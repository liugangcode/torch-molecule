from torch.utils.data import Dataset
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
import numpy as np

class MolGraphDataset(Dataset):
    """
    Dataset for MolGAN that converts SMILES to graph tensors.
    Outputs:
    - adj: [Y, N, N]
    - node: [N, T]
    - reward: float (optional)
    """

    def __init__(self, smiles_list, atom_types, bond_types, max_nodes=9, rewards=None):
        """
        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        atom_types : List[str]
            Ordered list of allowed atom types (e.g., ['C', 'O', 'N', 'F'])

        bond_types : List[int]
            List of allowed bond types (RDKit enums: SINGLE=1, DOUBLE=2, etc.)

        max_nodes : int
            Max number of atoms in any molecule (pad or skip otherwise)

        rewards : Optional[List[float]]
            Precomputed rewards (e.g., QED values)
        """
        self.smiles_list = smiles_list
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_nodes = max_nodes
        self.rewards = rewards if rewards is not None else [0.0] * len(smiles_list)

        self.atom_type_map = {atom: i for i, atom in enumerate(atom_types)}
        self.bond_type_map = {b: i for i, b in enumerate(bond_types)}

        self.data = [self._smiles_to_graph(s) for s in smiles_list]

    def _smiles_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()

        if num_atoms > self.max_nodes:
            raise ValueError(f"Too many atoms in molecule: {num_atoms} > {self.max_nodes}")

        # Node features
        node = np.zeros((self.max_nodes, len(self.atom_types)))
        for i, atom in enumerate(mol.GetAtoms()):
            atom_type = atom.GetSymbol()
            if atom_type not in self.atom_type_map:
                continue
            node[i, self.atom_type_map[atom_type]] = 1

        # Adjacency tensor
        adj = np.zeros((len(self.bond_types), self.max_nodes, self.max_nodes))
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = int(bond.GetBondTypeAsDouble())
            if bond_type not in self.bond_type_map:
                continue
            k = self.bond_type_map[bond_type]
            adj[k, i, j] = 1
            adj[k, j, i] = 1  # symmetric

        return {"adj": adj, "node": node}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        adj = torch.tensor(sample["adj"], dtype=torch.float32)
        node = torch.tensor(sample["node"], dtype=torch.float32)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        return {"adj": adj, "node": node, "reward": reward}
