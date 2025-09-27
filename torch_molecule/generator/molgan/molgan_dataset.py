import torch
from torch.utils.data import Dataset
from rdkit import Chem
from typing import List


class MolGANDataset(Dataset):
    """
    PyTorch Dataset class for MolGAN model, specifically dealing with converting
    SMILES data to Graph tensor data, which is suitable for MolGAN training.
    """

    def __init__(
        self,
        data: List[str],
        atom_types: List[str],
        bond_types: List[str],
        max_num_atoms: int = 50,
    ):
        """
        Initialize the MolGANDataset.

        Parameters
        ----------
        data : list of str
            List of SMILES strings representing the molecules.
        atom_types : list of str
            List of allowed atom types.
        bond_types : list of str
            List of allowed bond types.
        max_num_atoms : int
            Maximum number of atoms in a molecule for padding purposes.
        """
        self.data = data
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_num_atoms = max_num_atoms
        self.atom_type_to_idx = {atom: idx for idx, atom in enumerate(atom_types)}
        self.bond_type_to_idx = {bond: idx for idx, bond in enumerate(bond_types)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string at index {idx}: {smiles}")

        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.max_num_atoms:
            raise ValueError(f"Molecule at index {idx} exceeds max_num_atoms: {num_atoms} > {self.max_num_atoms}")

        # Initialize node features and adjacency matrix
        node_features = torch.zeros((self.max_num_atoms, len(self.atom_types)), dtype=torch.float)
        adjacency_matrix = torch.zeros((self.max_num_atoms, self.max_num_atoms, len(self.bond_types)), dtype=torch.float)

        # Fill node features
        for i, atom in enumerate(mol.GetAtoms()):
            atom_type = atom.GetSymbol()
            if atom_type in self.atom_type_to_idx:
                node_features[i, self.atom_type_to_idx[atom_type]] = 1.0

        # Fill adjacency matrix
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = str(bond.GetBondType())
            if bond_type in self.bond_type_to_idx:
                adjacency_matrix[begin_idx, end_idx, self.bond_type_to_idx[bond_type]] = 1.0
                adjacency_matrix[end_idx, begin_idx, self.bond_type_to_idx[bond_type]] = 1.0

        return node_features, adjacency_matrix












