"""Train/test splitting utilities for molecular SMILES datasets.

Random splitting is an i.i.d. baseline. Scaffold splitting groups molecules by
Bemis-Murcko frameworks so that the same scaffold does not appear in both
splits. Butina splitting groups by Taylor-Butina clusters on Morgan fingerprints
(sparse Tanimoto neighbor graph). Size splitting holds out larger (or smaller)
molecules by heavy-atom count.

The second split returned by ``train_test_split`` is a holdout set. Pass it to
``fit(..., X_val, y_val)`` as validation data. A disjoint final test set requires
a later three-way split API.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .constant import SMILESDataset

_SUPPORTED_METHODS = ("random", "scaffold", "butina", "size")
_SIZE_DIRECTIONS = ("small_to_large", "large_to_small")
_SIZE_MODES = ("standard", "sizeshiftreg")
_BUTINA_RADIUS = 2
_BUTINA_FP_SIZE = 2048
_BUTINA_DEFAULT_CUTOFF = 0.65
_SIZESHIFTREG_TRAIN_FRACTION = 0.5
_SIZESHIFTREG_TEST_FRACTION = 0.1
_BUTINA_OOM_MESSAGE = (
    "Butina split ran out of memory on the full dataset. "
    "Do not subsample to work around a split failure; that changes what "
    "the split measures. QM9-scale data is expected to fit; for much "
    "larger libraries an optional chemfp backend may be added later."
)

_MORGAN_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=_BUTINA_RADIUS, fpSize=_BUTINA_FP_SIZE
)


def subsample(
    dataset: SMILESDataset,
    n: int,
    seed: int = 0,
) -> SMILESDataset:
    """Draw a random subset without replacement.

    Intended for local debugging and CI, not as a way to make structure-aware
    splits cheaper. Do not subsample a benchmark because Butina is slow or
    memory-heavy; that changes what the split measures.

    Parameters
    ----------
    dataset : SMILESDataset
        Input dataset.
    n : int
        Number of molecules to keep. Must be at least 1 and at most the dataset
        size.
    seed : int, default=0
        Random seed.

    Returns
    -------
    SMILESDataset
        Subsampled dataset. If ``n`` equals the dataset size, a copy with the
        original order is returned.
    """
    _check_dataset(dataset)
    n_total = len(dataset.data)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if n > n_total:
        raise ValueError(
            f"n={n} is larger than the dataset size ({n_total})."
        )
    if n == n_total:
        return _subset(dataset, list(range(n_total)))

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=n, replace=False)
    return _subset(dataset, indices.tolist())


def train_test_split(
    dataset: SMILESDataset,
    test_size: float = 0.2,
    method: str = "random",
    seed: int = 42,
    *,
    use_csk: bool = False,
    similarity_cutoff: float = _BUTINA_DEFAULT_CUTOFF,
    direction: str = "small_to_large",
    mode: str = "standard",
) -> Tuple[SMILESDataset, SMILESDataset]:
    """Split a SMILES dataset into train and holdout subsets.

    Parameters
    ----------
    dataset : SMILESDataset
        Input dataset.
    test_size : float, default=0.2
        Fraction of molecules requested for the holdout set. For scaffold and
        Butina splits the realized fraction can differ because whole groups are
        assigned together. Ignored when ``method="size"`` and
        ``mode="sizeshiftreg"``.
    method : {"random", "scaffold", "butina", "size"}, default="random"
        ``"random"`` is an i.i.d. baseline (often optimistic for molecules).
        ``"scaffold"`` holds out unseen Bemis-Murcko scaffolds.
        ``"butina"`` holds out unseen Taylor-Butina clusters (Morgan / Tanimoto).
        ``"size"`` holds out molecules by heavy-atom count.
    seed : int, default=42
        Random seed. Used by ``random``. Scaffold, Butina clustering, and size
        assignment are deterministic; ``seed`` is accepted for API stability
        and ignored.
    use_csk : bool, default=False
        Scaffold only. If True, generic cyclic skeletons are used (all atoms
        as carbon). If False, atom types are kept (RDKit default).
    similarity_cutoff : float, default=0.65
        Butina only. Tanimoto **similarity** threshold in ``(0, 1]``. Molecules
        with similarity at least this value are neighbors. This is not
        DeepChem's distance cutoff.
    direction : {"small_to_large", "large_to_small"}, default="small_to_large"
        Size only (``mode="standard"``). ``small_to_large`` puts smaller
        molecules in train and larger ones in holdout.
    mode : {"standard", "sizeshiftreg"}, default="standard"
        Size only. ``sizeshiftreg`` uses the SizeShiftReg protocol: smallest
        50% train, largest 10% holdout; the middle 40% is unused.

    Returns
    -------
    train, holdout : SMILESDataset
        The second dataset is a holdout split intended as validation data for
        ``fit`` / ``autofit``.
    """
    _check_dataset(dataset)
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown split method {method!r}. "
            f"Supported methods: {list(_SUPPORTED_METHODS)}."
        )
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    n = len(dataset.data)
    if n < 2:
        raise ValueError("Need at least 2 molecules to split a dataset.")

    if method == "random":
        idx_train, idx_test = _random_split(n, test_size, seed)
    elif method == "scaffold":
        groups = _scaffold_groups(dataset.data, use_csk=use_csk)
        idx_train, idx_test = _group_split(groups, test_size)
    elif method == "butina":
        if not 0.0 < similarity_cutoff <= 1.0:
            raise ValueError(
                f"similarity_cutoff must be in (0, 1], got {similarity_cutoff}."
            )
        groups = _butina_groups_or_oom(
            dataset.data, similarity_cutoff=similarity_cutoff
        )
        idx_train, idx_test = _group_split(groups, test_size)
    else:
        idx_train, idx_test = _size_split(
            dataset.data,
            test_size=test_size,
            direction=direction,
            mode=mode,
        )

    return _subset(dataset, idx_train), _subset(dataset, idx_test)


def _check_dataset(dataset: SMILESDataset) -> None:
    if not isinstance(dataset, SMILESDataset):
        raise TypeError(
            f"dataset must be a SMILESDataset, got {type(dataset).__name__}."
        )
    if not isinstance(dataset.data, list):
        raise TypeError("dataset.data must be a list of SMILES strings.")
    if dataset.target is not None:
        target = np.asarray(dataset.target)
        if target.shape[0] != len(dataset.data):
            raise ValueError(
                f"target has {target.shape[0]} rows but data has "
                f"{len(dataset.data)} molecules."
            )


def _random_split(
    n: int, test_size: float, seed: int
) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_test = min(max(n_test, 1), n - 1)
    idx_test = np.sort(perm[:n_test]).tolist()
    idx_train = np.sort(perm[n_test:]).tolist()
    return idx_train, idx_test


def _mols_from_smiles(smiles_list: Sequence[str]) -> List[Chem.Mol]:
    from ..utils.checker import MolecularInputChecker

    invalid = []
    mols: List[Optional[Chem.Mol]] = [None] * len(smiles_list)
    for i, smiles in enumerate(smiles_list):
        if not isinstance(smiles, str):
            invalid.append(f"Non-string SMILES at index {i}: {smiles!r}")
            continue
        is_valid, error_msg, mol = MolecularInputChecker.validate_smiles(
            smiles, i
        )
        if not is_valid:
            invalid.append(error_msg)
            continue
        mols[i] = mol

    if invalid:
        raise ValueError("Invalid SMILES found:\n" + "\n".join(invalid))
    return mols  # type: ignore[return-value]


def _scaffold_groups(
    smiles_list: Sequence[str], use_csk: bool = False
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, mol in enumerate(_mols_from_smiles(smiles_list)):
        groups[_scaffold_key(mol, use_csk=use_csk)].append(i)
    return dict(groups)


def _scaffold_key(mol: Chem.Mol, use_csk: bool = False) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol)

    if use_csk:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(scaffold)


def _butina_groups(
    smiles_list: Sequence[str],
    similarity_cutoff: float = _BUTINA_DEFAULT_CUTOFF,
) -> Dict[str, List[int]]:
    clusters = _butina_clusters(smiles_list, similarity_cutoff=similarity_cutoff)
    return {f"cluster_{i}": members for i, members in enumerate(clusters)}


def _butina_groups_or_oom(
    smiles_list: Sequence[str],
    similarity_cutoff: float = _BUTINA_DEFAULT_CUTOFF,
) -> Dict[str, List[int]]:
    try:
        return _butina_groups(
            smiles_list, similarity_cutoff=similarity_cutoff
        )
    except MemoryError as exc:
        raise MemoryError(_BUTINA_OOM_MESSAGE) from exc


def _butina_clusters(
    smiles_list: Sequence[str],
    similarity_cutoff: float = _BUTINA_DEFAULT_CUTOFF,
) -> List[List[int]]:
    """Exact Taylor-Butina clusters via a sparse Tanimoto neighbor graph.

    Matches RDKit ``Butina.ClusterData`` membership (``reordering=False``)
    without storing the condensed distance matrix. Neighbor edges are pairs
    with Tanimoto similarity >= ``similarity_cutoff``. Degree includes self,
    matching RDKit's zero self-distance. Ties break like RDKit: higher index
    first among equal degrees.
    """
    mols = _mols_from_smiles(smiles_list)
    fps = [_MORGAN_FP_GEN.GetFingerprint(mol) for mol in mols]
    neighbor_lists = _butina_neighbor_lists(fps, similarity_cutoff)
    return _butina_exclusion_spheres(neighbor_lists)


def _butina_neighbor_lists(
    fps: Sequence[DataStructs.ExplicitBitVect],
    similarity_cutoff: float,
) -> List[List[int]]:
    n = len(fps)
    neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        neighbors[i].append(i)
        if i == 0:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        if sims:
            hits = np.flatnonzero(np.asarray(sims, dtype=np.float64) >= similarity_cutoff)
            for j in hits.tolist():
                neighbors[i].append(int(j))
                neighbors[j].append(i)
    for i in range(n):
        neighbors[i].sort()
    return neighbors


def _butina_exclusion_spheres(
    neighbor_lists: Sequence[Sequence[int]],
) -> List[List[int]]:
    n = len(neighbor_lists)
    sorted_indices = [
        (len(nbrs), idx) for idx, nbrs in enumerate(neighbor_lists)
    ]
    sorted_indices.sort(reverse=True)

    clusters: List[List[int]] = []
    seen = np.zeros(n, dtype=bool)

    while sorted_indices and sorted_indices[0][0] > 1:
        _, idx = sorted_indices.pop(0)
        if seen[idx]:
            continue
        cluster = [idx]
        seen[idx] = True
        for neighbor in neighbor_lists[idx]:
            if not seen[neighbor]:
                cluster.append(neighbor)
                seen[neighbor] = True
        clusters.append(cluster)

    while sorted_indices:
        _, idx = sorted_indices.pop(0)
        if seen[idx]:
            continue
        clusters.append([idx])
    return clusters


def _size_split(
    smiles_list: Sequence[str],
    test_size: float,
    direction: str = "small_to_large",
    mode: str = "standard",
) -> Tuple[List[int], List[int]]:
    if direction not in _SIZE_DIRECTIONS:
        raise ValueError(
            f"Unknown size direction {direction!r}. "
            f"Supported directions: {list(_SIZE_DIRECTIONS)}."
        )
    if mode not in _SIZE_MODES:
        raise ValueError(
            f"Unknown size mode {mode!r}. "
            f"Supported modes: {list(_SIZE_MODES)}."
        )

    mols = _mols_from_smiles(smiles_list)
    n_atoms = np.array([mol.GetNumHeavyAtoms() for mol in mols], dtype=np.int64)
    order = np.argsort(n_atoms, kind="stable")
    n = len(order)

    if mode == "sizeshiftreg":
        n_train = int(round(_SIZESHIFTREG_TRAIN_FRACTION * n))
        n_test = int(round(_SIZESHIFTREG_TEST_FRACTION * n))
        n_train = min(max(n_train, 1), n - 1)
        n_test = min(max(n_test, 1), n - n_train)
        idx_train = order[:n_train].tolist()
        idx_test = order[-n_test:].tolist()
        return idx_train, idx_test

    n_test = int(round(n * test_size))
    n_test = min(max(n_test, 1), n - 1)
    if direction == "small_to_large":
        idx_train = order[:-n_test].tolist()
        idx_test = order[-n_test:].tolist()
    else:
        idx_train = order[n_test:].tolist()
        idx_test = order[:n_test].tolist()
    return idx_train, idx_test


def _group_split(
    groups: Dict[str, List[int]], test_size: float
) -> Tuple[List[int], List[int]]:
    """Assign whole groups with DeepChem-style greedy filling.

    Groups are sorted by decreasing size (group id as a tie-break). A group
    goes to train if it still fits under the train cutoff; otherwise it goes to
    the holdout set.
    """
    n = sum(len(idx) for idx in groups.values())
    train_cutoff = (1.0 - test_size) * n
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))

    idx_train: List[int] = []
    idx_test: List[int] = []
    for _, members in ordered:
        members_sorted = sorted(members)
        if len(idx_train) + len(members_sorted) > train_cutoff:
            idx_test.extend(members_sorted)
        else:
            idx_train.extend(members_sorted)

    if not idx_train or not idx_test:
        raise ValueError(
            "Group split produced an empty train or holdout set. "
            "Try a different test_size or a dataset with more diverse groups."
        )
    return idx_train, idx_test


def _subset(dataset: SMILESDataset, indices: Sequence[int]) -> SMILESDataset:
    indices = list(indices)
    data = [dataset.data[i] for i in indices]
    if dataset.target is None:
        target: Optional[np.ndarray] = None
    else:
        target = np.asarray(dataset.target)[indices]
        if target.ndim == 1:
            target = target.reshape(-1, 1)
    return SMILESDataset(data=data, target=target)
