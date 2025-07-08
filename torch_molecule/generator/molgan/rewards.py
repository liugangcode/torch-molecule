import torch
import torch.nn as nn
from typing import List, Optional, Union
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors
from .gan_utils import RelationalGCNLayer, molgan_graph_from_smiles
from ...utils.graph.graph_from_smiles import graph_from_smiles


# Non-Neural reward functions based on RDKit
def qed_reward(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return QED.qed(mol) if mol else 0.0

def logp_reward(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return Crippen.MolLogP(mol) if mol else 0.0

def weight_reward(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcExactMolWt(mol) if mol else 0.0

def combo_reward(smiles: str, weights=(0.7, 0.3)) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    qed_score = QED.qed(mol)
    logp_score = Crippen.MolLogP(mol)
    return weights[0] * qed_score + weights[1] * logp_score

class RewardOracle:
    def __init__(self, kind="qed"):
        if kind == "qed":
            self.func = qed_reward
        elif kind == "logp":
            self.func = logp_reward
        elif kind == "combo":
            self.func = lambda s: combo_reward(s, weights=(0.6, 0.4))
        else:
            raise ValueError(f"Unknown reward type: {kind}")

    def __call__(self, smiles: str) -> float:
        return self.func(smiles)





# Reward Network using Relational GCNs
class RewardNeuralNetwork(nn.Module):
    """
    Reward Network that predicts reward from (adj, node) graphs.
    """

    def __init__(self, num_atom_types=5, num_bond_types=4, hidden_dim=128, num_layers=2, num_nodes=9):
        super().__init__()
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(RelationalGCNLayer(num_atom_types, hidden_dim, num_bond_types))

        for _ in range(1, num_layers):
            self.gcn_layers.append(RelationalGCNLayer(hidden_dim, hidden_dim, num_bond_types))

        self.readout = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, adj, node):
        """
        adj: [B, Y, N, N]
        node: [B, N, T]
        """
        h = node
        for layer in self.gcn_layers:
            h = layer(adj, h)

        h = h.view(h.size(0), -1)
        return self.readout(h).squeeze(-1)


def fit_reward_network(
    reward_model: RewardNeuralNetwork,
    train_loader,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    verbose: bool = True
):
    """
    Train the reward model to approximate oracle rewards.

    Parameters
    ----------
    reward_model : RewardNeuralNetwork
        The neural network to train

    train_loader : DataLoader
        Yields batches of (adj, node, reward)

    epochs : int
        Number of training epochs

    lr : float
        Learning rate

    weight_decay : float
        Optional L2 regularization

    device : str
        Device to run on ("cpu" or "cuda")

    verbose : bool
        Whether to print losses
    """
    model = reward_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_losses = []

        for batch in train_loader:
            adj = batch["adj"].to(device)      # [B, Y, N, N]
            node = batch["node"].to(device)    # [B, N, T]
            reward = batch["reward"].to(device)  # [B]

            pred = model(adj, node)  # [B]
            loss = criterion(pred, reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if verbose:
            print(f"[Epoch {epoch+1}/{epochs}] RewardNet Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")





# Combined reward wrapper: which uses either neural or oracle rewards
class RewardNetwork:
    """
    Combined reward network that can use either neural or oracle rewards.
    """

    def __init__(self, kind: str = "qed", num_atom_types=5, num_bond_types=4, hidden_dim=128, num_layers=2, num_nodes=9):
        if kind in ["qed", "logp", "combo"]:
            self.oracle = RewardOracle(kind)
            self.neural = None
        else:
            self.oracle = None
            self.neural = RewardNeuralNetwork(num_atom_types, num_bond_types, hidden_dim, num_layers, num_nodes)

    def train_neural(self, train_loader, epochs=10, lr=1e-3, weight_decay=0.0, device="cpu", verbose=True):
        """
        Train the neural reward network using the provided DataLoader.
        """
        if self.neural is None:
            raise ValueError("No neural network defined. Use an oracle reward instead.")

        fit_reward_network(
            self.neural,
            train_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose
        )


    def default_converter(self, smiles: str) -> tuple:
        try:
            graph = molgan_graph_from_smiles(
                smiles,
                atom_vocab=["C", "N", "O", "F"],
                bond_types=[1.0, 1.5, 2.0, 3.0],
                max_nodes=9
            )

            if graph is None:
                return None, None

            adj = torch.tensor(graph["adj"], dtype=torch.float32).unsqueeze(0)
            node = torch.tensor(graph["node"], dtype=torch.float32).unsqueeze(0)
            return adj, node
        except Exception as e:
            print(f"[RewardNetwork] SMILES conversion failed: {smiles} → {e}")
            return None, None

    def __call__(self, smiles: str) -> float:
        if self.oracle:
            return self.oracle(smiles)
        elif self.neural:
            # Convert SMILES to graph representation and pass through neural network
            adj, node = self.default_converter(smiles)
            if adj is not None and node is not None:
                return self.neural(adj, node).item()
            else:
                return 0.0
        else:
            raise ValueError("No valid reward function defined.")
