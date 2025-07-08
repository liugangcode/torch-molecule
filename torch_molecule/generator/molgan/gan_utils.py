from typing import Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem




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
