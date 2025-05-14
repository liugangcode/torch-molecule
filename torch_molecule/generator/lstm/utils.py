from rdkit import Chem
from typing import Set, Any, Optional
import torch
from .smiles_char_dict import SmilesCharDictionary

def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.
    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string
    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    if smiles is None:
        return None

    if len(smiles) > 0:
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
        else:
            return None
    else:
        return None

def remove_duplicates(list_with_duplicates):
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.
    Args:
        list_with_duplicates: list that possibly contains duplicates
    Returns:
        A list with no duplicates.
    """
    unique_set: Set[Any] = set()
    unique_list = []
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)

    return unique_list


def rnn_start_token_vector(batch_size, device='cpu'):
    """
    Returns a vector of start tokens for LSTM.
    This vector can be used to start sampling a batch of SMILES strings.

    Args:
        batch_size: how many SMILES will be generated at the same time in LSTM
        device: cpu | cuda

    Returns:
        a tensor (batch_size x 1) containing the start token
    """
    sd = SmilesCharDictionary()
    return torch.LongTensor(batch_size, 1).fill_(sd.begin_idx).to(device)
