import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from .features import atom_to_feature_vector, bond_to_feature_vector
from .features import getmaccsfingerprint, getmorganfingerprint
from ..generic.pseudo_tasks import PSEUDOTASK

def add_fingerprint_feature(mol, feature_type, get_fingerprint_fn):
    if feature_type is None:
        return None
    fingerprint = get_fingerprint_fn(mol)
    return np.expand_dims(np.array(fingerprint, dtype="int8"), axis=0)
    
def get_augmented_property(mol, properties):
    if mol is None:
        return None
            
    supported_properties = set(PSEUDOTASK.keys())
    unsupported = set(properties) - supported_properties
    if unsupported:
        raise ValueError(f"Unsupported properties: {unsupported}. Supported properties are: {supported_properties}")
    
    augmented_property = []
    if 'maccs' in properties:
        maccs = getmaccsfingerprint(mol)
        augmented_property.extend(maccs)
    if 'morgan' in properties:
        mgf = getmorganfingerprint(mol)
        augmented_property.extend(mgf)
    if 'logP' in properties:
        logp = Crippen.MolLogP(mol)
        augmented_property.append(logp)
    return augmented_property

def graph_from_smiles(smiles_or_mol, properties, augmented_features=None, augmented_properties=None):
    """
    Converts SMILES string or RDKit molecule to graph Data object
    
    Parameters
    ----------
    smiles_or_mol : Union[str, rdkit.Chem.rdchem.Mol]
        SMILES string or RDKit molecule object
    properties : Any
        Properties to include in the graph
    augmented_features : list
        List of augmented features to include
    augmented_properties : list, optional
        List of augmented properties to include
        
    Returns
    -------
    dict
        Graph object dictionary
    """
    # try:
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol
        
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        # print(atom.GetSymbol(), atom_to_feature_vector(atom)[0])
        atom_features_list.append(atom_to_feature_vector(atom))

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

    # Handle properties and augmented properties
    props_list = []            
    if properties is not None:
        props_list.append(np.array(properties, dtype=np.float32))
    if augmented_properties is not None:
        aug_props = get_augmented_property(mol, augmented_properties)
        if aug_props:
            props_list.append(np.array(aug_props, dtype=np.float32))
    if props_list:
        combined_props = np.concatenate(props_list)
        graph['y'] = combined_props.reshape(1, -1)
    else:
        graph['y'] = np.full((1, 1), np.nan, dtype=np.float32)

    # Handle augmented features
    if augmented_features is not None:
        graph['morgan'] = add_fingerprint_feature(
            mol,
            'morgan' if 'morgan' in augmented_features else None,
            getmorganfingerprint
        )
        graph['maccs'] = add_fingerprint_feature(
            mol,
            'maccs' if 'maccs' in augmented_features else None,
            getmaccsfingerprint
        )
    else:
        graph['morgan'] = None
        graph['maccs'] = None

    return graph    
    
    # except Exception as e:
    #     print(f"Error: {e} during converting {smiles_or_mol} to graph")
    #     return None
