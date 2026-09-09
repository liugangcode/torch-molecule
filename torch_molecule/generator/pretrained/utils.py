"""Representation conversion utilities for HF pretrained generators."""

import warnings
from typing import List

from rdkit import Chem


def _require_selfies():
    try:
        import selfies  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'selfies' package is required for SELFIES conversion. "
            "Install it with `pip install selfies`."
        ) from exc


def smiles_to_selfies(smiles: List[str]) -> List[str]:
    """Convert SMILES strings to SELFIES representations.

    Parameters
    ----------
    smiles : List[str]
        Input SMILES strings.

    Returns
    -------
    List[str]
        SELFIES strings in the same order as the input.

    Raises
    ------
    ValueError
        If a SMILES string is invalid or cannot be encoded as SELFIES.
    """
    _require_selfies()
    import selfies as sf

    selfies_list: List[str] = []
    for idx, smiles_string in enumerate(smiles):
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {idx}: {smiles_string}")
        canonical = Chem.MolToSmiles(mol)
        try:
            selfies_list.append(sf.encoder(canonical))
        except sf.EncoderError as exc:
            raise ValueError(
                f"SMILES at index {idx} is RDKit-valid but not SELFIES-encodable: {smiles_string}"
            ) from exc
    return selfies_list


def selfies_to_smiles(selfies_list: List[str]) -> List[str]:
    """Convert SELFIES strings to canonical SMILES.

    Entries that cannot be decoded are dropped and a warning is emitted.
    Unexpected errors such as ``ImportError`` are not swallowed.

    Parameters
    ----------
    selfies_list : List[str]
        Input SELFIES strings.

    Returns
    -------
    List[str]
        Canonical SMILES strings for entries that decoded successfully.
    """
    _require_selfies()
    import selfies as sf

    smiles_list: List[str] = []
    n_dropped = 0
    for selfies_string in selfies_list:
        if not selfies_string or not str(selfies_string).strip():
            n_dropped += 1
            continue
        try:
            decoded = sf.decoder(selfies_string)
        except sf.DecoderError:
            n_dropped += 1
            continue
        mol = Chem.MolFromSmiles(decoded) if decoded else None
        if mol is None:
            n_dropped += 1
            continue
        smiles_list.append(Chem.MolToSmiles(mol))

    if n_dropped:
        warnings.warn(f"dropped {n_dropped} invalid SELFIES", stacklevel=2)
    return smiles_list


def _require_fragment_selfies():
    try:
        from fragment_selfies import FragmentSelfiesCodec  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'fragment-selfies' package is required for Fragment-SELFIES conversion. "
            "Install it with `pip install fragment-selfies`."
        ) from exc


def smiles_to_fragment_selfies(smiles: List[str]) -> List[str]:
    """Convert SMILES strings to Fragment-SELFIES representations."""
    _require_fragment_selfies()

    try:
        from molexar.data.converter import smiles_to_fragment_selfies as encode_one
    except ImportError as exc:
        raise ImportError(
            "Molexar conversion utilities require the 'molexar' package. "
            "Install it with `pip install git+https://github.com/fairydance/Molexar.git`."
        ) from exc

    return [encode_one(smiles_string, canonical=True) for smiles_string in smiles]


def fragment_selfies_to_smiles(fragment_selfies_list: List[str]) -> List[str]:
    """Convert Fragment-SELFIES strings to canonical SMILES.

    Entries that cannot be decoded are dropped and a warning is emitted.
    Unexpected errors such as ``ImportError`` are not swallowed.
    """
    _require_fragment_selfies()

    try:
        from molexar.data.converter import fragment_selfies_to_smiles as decode_one
    except ImportError as exc:
        raise ImportError(
            "Molexar conversion utilities require the 'molexar' package. "
            "Install it with `pip install git+https://github.com/fairydance/Molexar.git`."
        ) from exc

    smiles_list: List[str] = []
    n_dropped = 0
    for fragment_selfies in fragment_selfies_list:
        if not fragment_selfies or not str(fragment_selfies).strip():
            n_dropped += 1
            continue
        try:
            decoded = decode_one(fragment_selfies, canonical=True, ignore_errors=False)
        except (ValueError, TypeError):
            n_dropped += 1
            continue
        if not decoded:
            n_dropped += 1
            continue
        smiles_list.append(decoded)

    if n_dropped:
        warnings.warn(f"dropped {n_dropped} invalid Fragment-SELFIES", stacklevel=2)
    return smiles_list
