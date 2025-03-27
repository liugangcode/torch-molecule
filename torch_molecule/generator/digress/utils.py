import torch
import json
from rdkit import Chem
import numpy as np
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops

def to_dense(x, edge_index, edge_attr, batch, max_num_nodes=None):
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if max_num_nodes is None:
        max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask

def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E

class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor, categorical: bool = False):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if categorical:
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

def compute_dataset_info(smiles_or_mol_list, cache_path=None):
    pt = Chem.GetPeriodicTable()
    atom_name_list = []
    atom_count_list = []
    for i in range(2, 119):
        atom_name_list.append(pt.GetElementSymbol(i))
        atom_count_list.append(0)
    atom_name_list.append('*')
    atom_count_list.append(0)
    
    bond_count_list = [0, 0, 0, 0, 0]
    bond_type_to_index = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    tansition_E = np.zeros((118, 118, 5))

    n_atom_list = []
    n_bond_list = []
    n_atoms_per_mol_count = {}
    max_node = 0
    for i, sms_or_mol in enumerate(smiles_or_mol_list):
        if isinstance(sms_or_mol, str):
            mol = Chem.MolFromSmiles(sms_or_mol)
        else:
            mol = sms_or_mol

        n_atom = mol.GetNumHeavyAtoms()
        n_bond = mol.GetNumBonds()
        max_node = max(max_node, n_atom)
        
        n_atom_list.append(n_atom)
        n_bond_list.append(n_bond)

        # Count atoms per molecule size
        if n_atom in n_atoms_per_mol_count:
            n_atoms_per_mol_count[n_atom] += 1
        else:
            n_atoms_per_mol_count[n_atom] = 1
            
        cur_atom_count_arr = np.zeros(118)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'H':
                continue
            elif symbol == '*':
                atom_count_list[-1] += 1
                cur_atom_count_arr[-1] += 1
            else:
                atom_count_list[atom.GetAtomicNum()-2] += 1
                cur_atom_count_arr[atom.GetAtomicNum()-2] += 1
        
        tansition_E_temp = np.zeros((118, 118, 5))
        for bond in mol.GetBonds():
            start_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
            if start_atom.GetSymbol() == 'H' or end_atom.GetSymbol() == 'H':
                continue
            
            if start_atom.GetSymbol() == '*':
                start_index = 117
            else:
                start_index = start_atom.GetAtomicNum() - 2
            if end_atom.GetSymbol() == '*':
                end_index = 117
            else:
                end_index = end_atom.GetAtomicNum() - 2

            bond_type = bond.GetBondType()
            bond_index = bond_type_to_index[bond_type]
            bond_count_list[bond_index] += 2
    
    # Create n_atoms_per_mol array with proper size
    n_atoms_per_mol = [0] * (max_node + 1)
    for n_atom, count in n_atoms_per_mol_count.items():
        n_atoms_per_mol[n_atom] = count
    
    n_atoms_per_mol = np.array(n_atoms_per_mol) / np.sum(n_atoms_per_mol)
    n_atoms_per_mol = n_atoms_per_mol.tolist()

    atom_count_list = np.array(atom_count_list) / np.sum(atom_count_list)
    active_atoms = np.array(atom_name_list)[atom_count_list > 0]
    active_atoms = active_atoms.tolist()
    atom_count_list = atom_count_list.tolist()

    bond_count_list = np.array(bond_count_list) / np.sum(bond_count_list)
    bond_count_list = bond_count_list.tolist()


    node_types = torch.Tensor(atom_count_list)
    edge_types = torch.Tensor(bond_count_list)
    active_index = (node_types > 0).nonzero().squeeze()

    x_margins = node_types.float() / torch.sum(node_types)
    x_margins = x_margins[active_index]
    e_margins = edge_types.float() / torch.sum(edge_types)

    num_nodes_dist = DistributionNodes(torch.Tensor(n_atoms_per_mol))

    meta_dict = {
        'active_index': active_index,
        'x_margins': x_margins,
        'e_margins': e_margins,
        'atom_decoder': active_atoms,
        'num_nodes_dist': num_nodes_dist,
        'max_node': max_node, 
        }

    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(meta_dict, f)

    return meta_dict

class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p