from typing import Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from ...utils.graph.graph_to_smiles import graph_to_smiles
from ...utils.graph.graph_from_smiles import graph_from_smiles




class RelationalGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_relations)])
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, adj, h):
        """
        adj: [B, Y, N, N]
        h: [B, N, D]
        """
        out = 0
        for i in range(self.num_relations):
            adj_i = adj[:, i, :, :]
            h_i = self.linears[i](h)
            out += torch.bmm(adj_i, h_i)

        out = out + self.bias
        return F.relu(out)


def encode_smiles_to_graph(
    smiles: str,
    atom_vocab: list = ["C", "N", "O", "F"],
    bond_types: list = [1.0, 1.5, 2.0, 3.0],
    max_nodes: int = 9
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert a SMILES string into (adj, node) tensors.

    Parameters
    ----------
    smiles : str
        Input SMILES string

    atom_vocab : list of str
        List of valid atom types

    bond_types : list of float
        Allowed bond types (e.g., 1.0: single, 2.0: double)

    max_nodes : int
        Max number of atoms (graph will be padded)

    Returns
    -------
    adj : Tensor [Y, N, N]
        Multi-relational adjacency tensor

    node : Tensor [N, T]
        One-hot atom features
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_nodes:
        # raise ValueError(f"Invalid or oversized molecule: {smiles}")
        return None

    N = max_nodes
    T = len(atom_vocab)
    Y = len(bond_types)

    # Initialize node features
    node = np.zeros((N, T), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= N:
            break
        symbol = atom.GetSymbol()
        if symbol in atom_vocab:
            node[i, atom_vocab.index(symbol)] = 1.0

    # Initialize adjacency tensor
    adj = np.zeros((Y, N, N), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        btype = bond.GetBondTypeAsDouble()
        if btype in bond_types and i < N and j < N:
            k = bond_types.index(btype)
            adj[k, i, j] = 1.0
            adj[k, j, i] = 1.0  # undirected

    # Convert to torch.Tensor
    return torch.tensor(adj), torch.tensor(node)


def decode_smiles(
        adj: torch.Tensor,
        node: torch.Tensor,
        atom_decoder: list = ["C", "N", "O", "F"]
    ) -> list:
    """
    Convert a batch of (adj, node) tensors to SMILES strings.

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency tensor of shape [B, Y, N, N]

    node : torch.Tensor
        Node feature tensor of shape [B, N, T]

    atom_decoder : list of str
        Atom types in order of one-hot encoding indices

    Returns
    -------
    List[str or None]
        Decoded SMILES strings or None for invalid molecules
    """
    # Ensure tensors are detached and moved to CPU
    adj_np = adj.detach().cpu().numpy()
    node_np = node.detach().cpu().numpy()

    # Build molecule list
    molecule_list = list(zip(node_np, adj_np))

    # Decode into SMILES strings
    smiles_list = graph_to_smiles(molecule_list, atom_decoder)

    return smiles_list











def molgan_graph_from_smiles(smiles: str, atom_vocab: list, bond_types: list, max_nodes: int) -> Optional[dict]:
    """
    Convert SMILES to MolGAN-style (adjacency, node) graph.

    Parameters
    ----------
    smiles : str
        SMILES string

    atom_vocab : list of str
        List of allowed atom types (e.g., ['C', 'N', 'O', 'F'])

    bond_types : list of float
        List of bond types (e.g., [1.0, 1.5, 2.0, 3.0])

    max_nodes : int
        Maximum number of atoms

    Returns
    -------
    dict with keys:
        'adj': [Y, N, N] tensor
        'node': [N, T] tensor
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_nodes:
        return None

    T = len(atom_vocab)
    Y = len(bond_types)

    node = np.zeros((max_nodes, T))
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in atom_vocab:
            node[i, atom_vocab.index(symbol)] = 1

    adj = np.zeros((Y, max_nodes, max_nodes))
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        btype = bond.GetBondTypeAsDouble()
        if btype in bond_types:
            k = bond_types.index(btype)
            adj[k, i, j] = 1
            adj[k, j, i] = 1

    return {
        "adj": torch.tensor(adj, dtype=torch.float32).unsqueeze(0),
        "node": torch.tensor(node, dtype=torch.float32).unsqueeze(0)
    }
