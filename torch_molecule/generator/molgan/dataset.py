from torch.utils.data import Dataset
import torch
from typing import List, Optional, Callable
from rdkit import Chem

from .rewards import RewardNetwork
from .gan_utils import encode_smiles_to_graph
from ...utils.graph.graph_from_smiles import graph_from_smiles


class MolGraphDataset(Dataset):
    """
    Dataset for MolGAN: converts SMILES strings to graph format.

    Outputs a dict with:
    - 'adj': [Y, N, N] adjacency tensor
    - 'node': [N, T] node feature matrix
    - 'reward': float (optional)
    - 'smiles': original SMILES (optional)
    """

    def __init__(self,
                 smiles_list: List[str],
                 reward_function: Optional[RewardNetwork] = None,
                 max_nodes: int = 9,
                 drop_invalid: bool = True):
        """
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to convert into graph format.

        reward_function : Callable[[str], float], optional
            If provided, computes a scalar reward per molecule (e.g., QED, logP).
            Must accept a SMILES string and return a float.

        max_nodes : int
            Maximum allowed number of atoms (molecules exceeding this are dropped).

        drop_invalid : bool
            Whether to skip invalid or unparsable SMILES.
        """
        self.samples = []

        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError("Invalid SMILES")

                if mol.GetNumAtoms() > max_nodes:
                    raise ValueError("Too many atoms")

                # Compute reward if needed
                graph = encode_smiles_to_graph(smiles)
                if graph is None:
                    raise ValueError("Failed to encode SMILES to graph")
                adj, node = graph
                reward = reward_function(node, adj) if reward_function else 0.0

                # Convert to graph
                graph = graph_from_smiles(smiles, properties=reward)

                # Sanity check
                if 'adj' not in graph or 'node' not in graph:
                    raise ValueError("Incomplete graph data")

                graph['reward'] = reward
                graph['smiles'] = smiles
                self.samples.append(graph)

            except Exception as e:
                if not drop_invalid:
                    self.samples.append({
                        "adj": torch.zeros(1, max_nodes, max_nodes),
                        "node": torch.zeros(max_nodes, 1),
                        "reward": 0.0,
                        "smiles": smiles
                    })
                else:
                    print(f"[MolGraphDataset] Skipping SMILES {smiles}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "adj": torch.tensor(sample["adj"], dtype=torch.float32),
            "node": torch.tensor(sample["node"], dtype=torch.float32),
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "smiles": sample["smiles"]
        }

def molgan_collate_fn(batch):
    adj = torch.stack([item["adj"] for item in batch], dim=0)
    node = torch.stack([item["node"] for item in batch], dim=0)
    reward = torch.stack([item["reward"] for item in batch], dim=0)
    smiles = [item["smiles"] for item in batch]
    return {"adj": adj, "node": node, "reward": reward, "smiles": smiles}

