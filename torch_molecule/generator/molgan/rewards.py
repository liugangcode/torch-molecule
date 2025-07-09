from typing import Optional
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors
from .gan_utils import RelationalGCNLayer, molgan_graph_from_smiles
from ...utils.graph.graph_to_smiles import graph_to_smiles


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

    def __init__(self,
                 num_atom_types=5,
                 num_bond_types=4,
                 hidden_dim=128,
                 num_layers=2,
                 num_nodes=9):
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
            adj = batch["adj"].to(device)
            node = batch["node"].to(device)
            reward = batch["reward"].to(device)

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
    Combined reward network that uses either a neural model or an oracle.
    Accepts (adj, node) tensors as standard input.
    """

    def __init__(
            self,
            kind="qed",
            reward_net: Optional[RewardNeuralNetwork] = None,
            atom_decoder=None,
            device="cpu"):
        self.kind = kind
        self.device = device
        self.atom_decoder = atom_decoder or ["C", "N", "O", "F"]

        if kind in ["qed", "logp", "combo"]:
            self.oracle = RewardOracle(kind)
            self.neural = None
            if reward_net is not None:
                raise ValueError("reward_net should not be provided for oracle modes")
        elif kind == "neural":
            assert reward_net is not None, "reward_net must be provided for 'neural' mode"
            self.oracle = None
            self.neural = reward_net.to(device).eval()
        else:
            raise ValueError(f"Invalid kind: {kind}")

    def __call__(self, adj: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        """
        Compute reward from graph tensors.

        Parameters
        ----------
        adj : Tensor [B, Y, N, N]
        node : Tensor [B, N, T]

        Returns
        -------
        Tensor [B] : reward per sample
        """
        if self.neural is not None:
            with torch.no_grad():
                return self.neural(adj.to(self.device), node.to(self.device))

        elif self.oracle:
            graphs = list(zip(node.cpu().numpy(), adj.cpu().numpy()))
            smiles_list = graph_to_smiles(graphs, self.atom_decoder)
            rewards = [self.oracle(s) if s else 0.0 for s in smiles_list]
            return torch.tensor(rewards, dtype=torch.float32, device=self.device)

        else:
            raise ValueError("No reward function defined.")
