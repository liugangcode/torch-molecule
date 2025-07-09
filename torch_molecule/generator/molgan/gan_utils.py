from typing import Optional
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from ...utils.graph.graph_to_smiles import (
    build_molecule_with_partial_charges,
    correct_mol,
    mol2smiles,
    get_mol
)


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


ATOM_DECODER = ["C", "N", "O", "F"]  # Adjust based on your vocabulary
BOND_DICT = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def decode_smiles_from_graph(
    adj: torch.Tensor,
    node: torch.Tensor,
    atom_decoder: Optional[list] = ATOM_DECODER
) -> Optional[str]:
    """
    Converts (adj, node) graph back to a SMILES string.

    Parameters
    ----------
    adj : torch.Tensor
        Tensor of shape [Y, N, N] with binary bond type edges.
    node : torch.Tensor
        Tensor of shape [N, T] with atom type softmax/one-hot.
    atom_decoder : list
        List mapping indices to atom symbols.

    Returns
    -------
    Optional[str]
        SMILES string if successful, None otherwise.
    """
    try:
        atom_types = node.argmax(dim=-1)  # [N]
        edge_types = torch.argmax(adj, dim=0)  # [N, N], index of strongest bond type

        # Convert to RDKit Mol
        mol_init = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)

        # Try to correct connectivity and valency
        for connection in (True, False):
            mol_corr, _ = correct_mol(mol_init, connection=connection)
            if mol_corr is not None:
                break
        else:
            mol_corr = mol_init  # fallback

        # Final sanitization
        smiles = mol2smiles(mol_corr)
        if not smiles:
            smiles = Chem.MolToSmiles(mol_corr)

        # Canonicalize and return
        mol = get_mol(smiles)
        if mol is not None:
            frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            largest = max(frags, key=lambda m: m.GetNumAtoms())
            final_smiles = mol2smiles(largest)
            return final_smiles if final_smiles and len(final_smiles) > 1 else None
        return None

    except Exception as e:
        print(f"[MolGAN Decode] Error during decoding: {e}")
        return None



