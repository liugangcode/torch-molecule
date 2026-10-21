"""Deprecated per-dataset loaders, kept for one release.

Each calls :func:`torch_molecule.datasets.load_dataset` with the dataset's name in the
molecular dataset collection. Use ``load_dataset`` directly; these will be removed.
"""
import warnings
from typing import List, Optional

from .constant import SMILESDataset
from .hub import load_dataset

_RETURNS = """
    Returns
    -------
    dataset : SMILESDataset
        ``data`` is a list of canonical SMILES. ``target`` is a NumPy array of shape
        ``(n_molecules, n_targets)``, with ``NaN`` for missing measurements.
    local_dir : str, optional
        The cache directory, returned alongside ``dataset`` only when
        ``return_local_data_path=True``.
"""


def _returns(function):
    """Append the shared Returns section to a loader's docstring."""
    function.__doc__ = function.__doc__.rstrip() + "\n" + _RETURNS
    return function


def _alias(old: str, name: str, targets, local_dir: str, return_local_data_path: bool):
    warnings.warn(f"{old}() is deprecated and will be removed; use load_dataset({name!r}).",
                  FutureWarning, stacklevel=3)
    data = load_dataset(name, targets=targets, local_dir=local_dir)
    return (data, local_dir) if return_local_data_path else data


@_returns
def load_qm9(local_dir: str = "torchmol_data", target_cols: List[str] = ["gap"],
             return_local_data_path: bool = False) -> SMILESDataset:
    """Deprecated: use ``load_dataset("qm9", targets=["gap"])``.

    Parameters
    ----------
    local_dir : str, default="torchmol_data"
        Directory used to cache downloaded files. Existing files are reused.
    target_cols : list of str, default=['gap']
        Target columns to select, in the desired output order.
    return_local_data_path : bool, default=False
        Also return the cache directory.
    """
    return _alias("load_qm9", "qm9", target_cols, local_dir, return_local_data_path)


@_returns
def load_chembl2k(local_dir: str = "torchmol_data", target_cols: Optional[List[str]] = None,
                  return_local_data_path: bool = False) -> SMILESDataset:
    """Deprecated: use ``load_dataset("chembl2k")``.

    Parameters
    ----------
    local_dir : str, default="torchmol_data"
        Directory used to cache downloaded files. Existing files are reused.
    target_cols : list of str, optional
        Target columns to select; default all 41 targets.
    return_local_data_path : bool, default=False
        Also return the cache directory.
    """
    return _alias("load_chembl2k", "chembl2k", target_cols, local_dir, return_local_data_path)


@_returns
def load_broad6k(local_dir: str = "torchmol_data", target_cols: Optional[List[str]] = None,
                 return_local_data_path: bool = False) -> SMILESDataset:
    """Deprecated: use ``load_dataset("broad6k")``.

    Parameters
    ----------
    local_dir : str, default="torchmol_data"
        Directory used to cache downloaded files. Existing files are reused.
    target_cols : list of str, optional
        Assay columns to select; default all assays.
    return_local_data_path : bool, default=False
        Also return the cache directory.
    """
    return _alias("load_broad6k", "broad6k", target_cols, local_dir, return_local_data_path)


@_returns
def load_toxcast(local_dir: str = "torchmol_data", target_cols: Optional[List[str]] = None,
                 return_local_data_path: bool = False) -> SMILESDataset:
    """Deprecated: use ``load_dataset("toxcast")``.

    Parameters
    ----------
    local_dir : str, default="torchmol_data"
        Directory used to cache downloaded files. Existing files are reused.
    target_cols : list of str, optional
        Assay columns to select; default all 617 assays.
    return_local_data_path : bool, default=False
        Also return the cache directory.
    """
    return _alias("load_toxcast", "toxcast", target_cols, local_dir, return_local_data_path)


@_returns
def load_admet(local_dir: str = "torchmol_data", target_cols: Optional[List[str]] = None,
               return_local_data_path: bool = False) -> SMILESDataset:
    """Deprecated: use ``load_dataset("admet_ai")``.

    Parameters
    ----------
    local_dir : str, default="torchmol_data"
        Directory used to cache downloaded files. Existing files are reused.
    target_cols : list of str, optional
        Endpoints to select; default all 41 endpoints.
    return_local_data_path : bool, default=False
        Also return the cache directory.
    """
    return _alias("load_admet", "admet_ai", target_cols, local_dir, return_local_data_path)


@_returns
def load_zinc250k(local_dir: str = "torchmol_data", return_local_data_path: bool = False) -> SMILESDataset:
    """Deprecated: use ``load_dataset("zinc250k", targets=[])``; ``target`` is ``None``.

    Parameters
    ----------
    local_dir : str, default="torchmol_data"
        Directory used to cache downloaded files. Existing files are reused.
    return_local_data_path : bool, default=False
        Also return the cache directory.
    """
    return _alias("load_zinc250k", "zinc250k", [], local_dir, return_local_data_path)


@_returns
def load_gasperm(target_cols: List[str] = ["CH4", "CO2", "H2", "N2", "O2"],
                 local_dir: str = "torchmol_data") -> SMILESDataset:
    """Deprecated: use ``load_dataset("polymer_gas_permeability")``.

    Parameters
    ----------
    target_cols : list of str, default=['CH4', 'CO2', 'H2', 'N2', 'O2']
        Gases to select, in the desired output order.
    local_dir : str, default="torchmol_data"
        Directory used to cache downloaded files. Existing files are reused.
    """
    return _alias("load_gasperm", "polymer_gas_permeability", target_cols, local_dir, False)
