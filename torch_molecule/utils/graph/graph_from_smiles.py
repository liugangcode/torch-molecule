
import numpy as np
from rdkit import Chem
from .features import atom_to_feature_vector, bond_to_feature_vector

def graph_from_smiles(smiles_string, properties):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        # atoms
        atom_features_list = []
        # atom_label = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
            # atom_label.append(atom.GetSymbol())

        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)
        if properties is None:
            y = np.full(1, np.nan, dtype=np.float32)
            graph['y'] = y.reshape(1, -1)
        else:
            graph['y'] = np.array(properties, dtype=np.float32).reshape(1, -1)

        return graph
    except:
        return None