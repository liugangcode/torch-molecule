from typing import List, Optional, Callable
from rdkit import Chem
import torch
from torch.utils.data import Dataset
from .molgan_utils import qed_reward_fn

class MolGANDataset(Dataset):
    """
    A PyTorch Dataset for MolGAN, with all RDKit and graph tensor processing
    precomputed in __init__ for fast, pure-tensor __getitem__ access.
    Optionally caches property values for each molecule.
    """
    def __init__(
        self,
        data: List[str],
        atom_types: List[str],
        bond_types: List[str],
        max_num_atoms: int = 50,
        cache_properties: bool = False,
        property_fn: Optional[Callable] = None,
        return_mol: bool = False,
        device: Optional[torch.device] = None
    ):
        self.data = data
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_num_atoms = max_num_atoms
        self.atom_type_to_idx = {atom: idx for idx, atom in enumerate(atom_types)}
        self.bond_type_to_idx = {bond: idx for idx, bond in enumerate(bond_types)}
        self.return_mol = return_mol
        self.device = torch.device(device) if device is not None else None

        self.node_features = []
        self.adjacency_matrices = []
        self.mols = []
        self.cached_properties = [] if cache_properties and property_fn else None

        self.property_fn = property_fn if property_fn is not None else qed_reward_fn

        for idx, smiles in enumerate(self.data):
            mol = Chem.MolFromSmiles(smiles)
            self.mols.append(mol)
            # Default: if invalid, fill with zeros and (optionally) property 0
            nf = torch.zeros((self.max_num_atoms, len(self.atom_types)), dtype=torch.float)
            adj = torch.zeros((self.max_num_atoms, self.max_num_atoms, len(self.bond_types)), dtype=torch.float)
            prop_val = 0.0 if cache_properties else None

            if mol is not None:
                num_atoms = mol.GetNumAtoms()
                if num_atoms > self.max_num_atoms:
                    raise ValueError(f"Molecule at index {idx} exceeds max_num_atoms: {num_atoms} > {self.max_num_atoms}")

                for i, atom in enumerate(mol.GetAtoms()):
                    atom_type = atom.GetSymbol()
                    if atom_type in self.atom_type_to_idx:
                        nf[i, self.atom_type_to_idx[atom_type]] = 1.0

                for bond in mol.GetBonds():
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    bond_type = str(bond.GetBondType())
                    if bond_type in self.bond_type_to_idx:
                        bidx = self.bond_type_to_idx[bond_type]
                        adj[begin_idx, end_idx, bidx] = 1.0
                        adj[end_idx, begin_idx, bidx] = 1.0

                if cache_properties and self.property_fn:
                    try:
                        prop_val = self.property_fn(mol)
                    except Exception:
                        prop_val = 0.0

            # Move tensors to device immediately if a device is set
            if self.device is not None:
                nf = nf.to(self.device)
                adj = adj.to(self.device)

            self.node_features.append(nf)
            self.adjacency_matrices.append(adj)
            if cache_properties and property_fn and self.cached_properties is not None:
                self.cached_properties.append(prop_val)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        parts = [
            self.node_features[idx],
            self.adjacency_matrices[idx]
        ]
        # add optional property
        if self.cached_properties is not None:
            parts.append(self.cached_properties[idx])
        # add optional Mol object (can always access it if you want)
        if self.return_mol:
            parts.append(self.mols[idx])

        # Default: (node_features, adjacency_matrix)
        # With property: (node_features, adjacency_matrix, property)
        # With property and mol: (node_features, adjacency_matrix, property, mol)
        # With only mol: (node_features, adjacency_matrix, mol)
        return tuple(parts)







