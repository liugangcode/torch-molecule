"""RDKit molecule wrapper for the sand featurization pipeline."""
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from ..graph import MolecularGraph


class RDMolecule:
    """A thin wrapper around an RDKit molecule with featurization and utility methods."""

    def __init__(self, rdmol: Chem.Mol) -> None:
        """Args:
            rdmol: Underlying RDKit molecule object.
        """
        self.rdmol = rdmol
        self._canonical_smiles: str | None = None

    @classmethod
    def from_smiles(cls, smiles: str) -> "RDMolecule":
        """Construct an :class:`RDMolecule` from a SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            A new :class:`RDMolecule`.

        Raises:
            ValueError: If the SMILES string cannot be parsed.
        """
        rdmol = Chem.MolFromSmiles(smiles)
        if rdmol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return cls(rdmol)

    def to_graph(self, add_pos: bool = False) -> MolecularGraph:
        """Convert to a :class:`MolecularGraph` using the RDKit featurizer."""
        from .featurizer import RDKitGraphFeaturizer

        featurizer = RDKitGraphFeaturizer(add_pos=add_pos)
        return featurizer(self.rdmol)

    def to_smiles(self, ignoreAtomMapNumbers: bool = True) -> str:
        """Return the canonical SMILES string for this molecule.

        Args:
            ignoreAtomMapNumbers: Strip atom-map number annotations before
                canonicalization.

        Returns:
            Canonical SMILES string.
        """
        mol = Chem.Mol(self.rdmol)
        if ignoreAtomMapNumbers:
            for atom in mol.GetAtoms():
                atom.ClearProp("molAtomMapNumber")
        smiles = Chem.MolToSmiles(mol)
        return smiles
    
    def to_sd_string(self) -> str:
        """Return the molecule as an SD-file (MOL block) formatted string.

        Returns:
            MOL/SD block string.
        """
        sd_string = Chem.MolToMolBlock(self.rdmol)
        return sd_string

    @property
    def canonical_smiles(self) -> str:
        """Return the canonical SMILES representation."""
        if self._canonical_smiles is None:
            self._canonical_smiles = Chem.MolToSmiles(self.rdmol, canonical=True, ignoreAtomMapNumbers=True)
        return self._canonical_smiles
    
    @property
    def num_atoms(self) -> int:
        """Return the number of atoms in the molecule."""
        return self.rdmol.GetNumAtoms()
    
    @property
    def num_bonds(self) -> int:
        """Return the number of bonds in the molecule."""
        return self.rdmol.GetNumBonds()
    
    @property
    def num_conformers(self) -> int:
        """Return the number of conformers attached to the molecule."""
        return self.rdmol.GetNumConformers()

    @property
    def has_conformer(self) -> bool:
        """Check if the molecule has a 3D conformer attached."""
        return self.num_conformers > 0

    def __eq__(self, other: object) -> bool:
        """Compare two :class:`RDMolecule` objects by canonical SMILES.

        Raises:
            NotImplementedError: If either molecule has an embedded 3D conformer.
        """
        if not isinstance(other, RDMolecule):
            return NotImplemented
        if self.has_conformer or other.has_conformer:
            raise NotImplementedError(
                "Equality comparison for molecules with 3D conformers is not yet implemented"
            )
        return self.canonical_smiles == other.canonical_smiles

    def __hash__(self) -> int:
        """Hash by canonical SMILES.

        Raises:
            NotImplementedError: If the molecule has an embedded 3D conformer.
        """
        if self.has_conformer:
            raise NotImplementedError(
                "Hashing for molecules with 3D conformers is not yet implemented"
            )
        return hash(self.canonical_smiles)
    
    def sample_conformations(self, num_confs: int = 1, optimize: bool = False, num_threads: int = -1, prune_rms_thresh: float = 0.5) -> None:
        """Generate 3D conformations for the molecule.

        Args:
            num_confs (int): Number of conformations to generate.
            optimize (bool): Whether to optimize the generated conformations using MMFF94.
            num_threads (int): Number of threads to use for conformation generation. Default is -1, which uses all available threads.
            prune_rms_thresh (float): RMS threshold for pruning similar conformations. Only used if num_confs > 1.
        """
        # adding hydrogens
        self.rdmol = Chem.AddHs(self.rdmol)
        params = AllChem.ETKDGv3()  # type: ignore[attr-defined]  # rdkit-stubs is missing this (real rdkit.Chem.AllChem attribute)
        params.pruneRmsThresh = prune_rms_thresh
        if num_threads < 0:
            # use all available threads
            params.numThreads = 0
        elif num_threads == 0:
            # use single thread
            params.numThreads = 1
        else:
            params.numThreads = num_threads
        if num_confs > 1:
            conf_ids = AllChem.EmbedMultipleConfs(self.rdmol, numConfs=num_confs, params=params)  # type: ignore[attr-defined]
        else:
            conf_id = AllChem.EmbedMolecule(self.rdmol, params=params)  # type: ignore[attr-defined]
            conf_ids = [conf_id]
        if optimize:
            AllChem.MMFFOptimizeMoleculeConfs(self.rdmol, numThreads=num_threads)  # type: ignore[attr-defined]
        self.rdmol = Chem.RemoveHs(self.rdmol)

    def show_2d(self, size: Tuple[int, int] = (300, 300), remove_atom_map_numbers: bool = True):
        """Render the molecule as a 2D structure image.

        Args:
            size: ``(width, height)`` of the output image in pixels.
            remove_atom_map_numbers: Strip atom-map annotations before rendering.

        Returns:
            PIL :class:`Image` of the 2D depiction.
        """
        from rdkit.Chem import Draw  # lazy: keeps Pillow off the featurization path

        mol = Chem.Mol(self.rdmol)
        mol.RemoveAllConformers()
        if remove_atom_map_numbers:
            for atom in mol.GetAtoms():
                atom.ClearProp("molAtomMapNumber")
        return Draw.MolToImage(mol, size=size)
    
    def show_3d(self, size: Tuple[int, int] = (300, 300), remove_atom_map_numbers: bool = True):
        """Render the molecule using its 3D conformer.

        Args:
            size: ``(width, height)`` of the output image in pixels.
            remove_atom_map_numbers: Strip atom-map annotations before rendering.

        Returns:
            PIL :class:`Image` object.

        Raises:
            ValueError: If the molecule has no embedded conformer.
        """
        from rdkit.Chem import Draw  # lazy: keeps Pillow off the featurization path

        mol = Chem.Mol(self.rdmol)
        if remove_atom_map_numbers:
            for atom in mol.GetAtoms():
                atom.ClearProp("molAtomMapNumber")
        if not self.has_conformer:
            raise ValueError("Molecule does not have a 3D conformer to show")
        return Draw.MolToImage(mol, size=size)
    
    def show(self, size: Tuple[int, int] = (300, 300)):
        """Render the molecule as a 2D structure image.

        Args:
            size: ``(width, height)`` of the output image in pixels.

        Returns:
            PIL :class:`Image` of the 2D depiction.
        """
        return self.show_2d(size=size)
    
    def __repr__(self) -> str:
        return f"RDMolecule(SMILES: {self.to_smiles()})"


