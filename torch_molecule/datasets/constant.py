from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class SMILESDataset:
    """
    Data class for storing molecular SMILES dataset with input and target data.

    Attributes:
        data (List[str]): Input data (e.g., list of SMILES strings)
        target (np.ndarray | None): Target property values as 2D numpy array (rows=molecules, cols=targets) or None
        target_names (List[str] | None): Name of each target column, when known
        groups (List[str] | None): Structure key per row; rows with the same key are the same molecule
        info (dict): Provenance, units, label kinds, splits, and quality tags for hub datasets
        extra (Dict[str, list] | None): Other per-row source columns, such as conditions or dates
    """
    data: List[str]
    target: np.ndarray | None
    target_names: Optional[List[str]] = None
    groups: Optional[List[str]] = None
    info: dict = field(default_factory=dict)
    extra: Optional[Dict[str, list]] = None

    def subsample(self, n: int, seed: int = 0) -> "SMILESDataset":
        """Draw a random subset without replacement.

        Intended for local debugging and CI, not as a way to make structure-
        aware splits cheaper. Do not subsample a benchmark (for example QM9)
        because Butina is slow or memory-heavy; that changes what the split
        measures.

        Parameters
        ----------
        n : int
            Number of molecules to keep.
        seed : int, default=0
            Random seed.
        """
        from .split import subsample

        return subsample(self, n=n, seed=seed)

    def train_test_split(
        self,
        test_size: float = 0.2,
        method: str = "random",
        seed: int = 42,
        **kwargs,
    ) -> Tuple["SMILESDataset", "SMILESDataset"]:
        """Split into train and holdout ``SMILESDataset`` objects.

        Parameters
        ----------
        test_size : float, default=0.2
            Requested holdout fraction.
        method : {"random", "scaffold", "butina", "size"}, default="random"
            ``random`` assigns individual rows, so duplicate structures can
            appear in both splits. ``groups`` metadata is not used to assign
            rows. ``scaffold`` holds out unseen Bemis-Murcko scaffolds;
            ``butina`` holds out
            unseen Taylor-Butina clusters; ``size`` splits by heavy-atom count.
        seed : int, default=42
            Random seed (used by ``random``).
        **kwargs
            Extra options forwarded to the splitter (``use_csk`` for scaffold,
            ``similarity_cutoff`` for butina, ``direction`` / ``mode`` for size).

        Returns
        -------
        train, holdout : SMILESDataset
            The second dataset is intended as validation data for ``fit``.
        """
        from .split import train_test_split

        return train_test_split(
            self, test_size=test_size, method=method, seed=seed, **kwargs
        )
