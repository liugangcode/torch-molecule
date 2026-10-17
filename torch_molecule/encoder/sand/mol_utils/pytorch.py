"""Datasets and batching helpers for molecular graphs."""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from .graph import MolecularGraph
from .rdkit.molecule import RDMolecule


def smiles_to_batch(smiles_list: list[str]) -> MolecularGraph:
    """Convert a list of SMILES strings to a batched :class:`MolecularGraph`.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        A batched :class:`MolecularGraph` ready for model input.
    """
    mol_list = [RDMolecule.from_smiles(sml) for sml in smiles_list]
    graph_list = [mol.to_graph() for mol in mol_list]
    batch = MolecularGraph.from_graph_list(graph_list)
    return batch


class MolecularGraphDataset(Dataset):
    """A :class:`torch.utils.data.Dataset` that featurizes SMILES strings on the fly."""

    def __init__(self, smiles: list[str]) -> None:
        """Args:
            smiles: List of SMILES strings.
        """
        self.smiles = smiles

    def __len__(self) -> int:
        return len(self.smiles)

    def sml_to_graph(self, sml: str) -> MolecularGraph:
        """Convert a SMILES string to a :class:`MolecularGraph` with the SMILES stored as an attribute.

        Args:
            sml: SMILES string.

        Returns:
            :class:`MolecularGraph` with a ``smiles`` attribute.
        """
        mol = RDMolecule.from_smiles(sml)
        graph = mol.to_graph()
        graph["smiles"] = sml
        return graph

    def __getitem__(self, idx: int) -> MolecularGraph:
        """Return the featurized graph for the molecule at index ``idx``.

        Args:
            idx: Dataset index.

        Returns:
            :class:`MolecularGraph` with a ``smiles`` attribute.
        """
        sml = self.smiles[idx]
        graph = self.sml_to_graph(sml)
        return graph

    def collate_fn(self, data_list: list[MolecularGraph]) -> MolecularGraph:
        """Collate a list of :class:`MolecularGraph` objects into a single batched graph.

        Args:
            data_list: List of individual :class:`MolecularGraph` objects.

        Returns:
            A batched :class:`MolecularGraph`.
        """
        batch = MolecularGraph.from_graph_list(data_list)
        return batch


class MolecularGraphDataloader(DataLoader):
    """A :class:`torch.utils.data.DataLoader` that yields batched :class:`MolecularGraph` objects for a list of SMILES."""

    def __init__(
        self,
        smiles_list: list[str],
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> None:
        """Args:
            smiles_list: List of SMILES strings to iterate over.
            batch_size: Number of molecules per batch.
            shuffle: Whether to shuffle the dataset each epoch.
            num_workers: Number of DataLoader worker processes.
        """
        dataset = MolecularGraphDataset(smiles_list)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
        )
