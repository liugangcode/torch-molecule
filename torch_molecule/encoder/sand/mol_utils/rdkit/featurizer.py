"""RDKit-based molecular graph featurizer.

Produces a :class:`~torch_molecule.encoder.sand.mol_utils.graph.MolecularGraph` with a fixed tensor
schema:

* ``x``          - node features, shape ``(num_atoms, 16)``
* ``edge_index`` - shape ``(2, num_edges)`` (both directions, sorted)
* ``edge_attr``  - shape ``(num_edges, 5)``
* ``pos``        - optional, shape ``(num_atoms, num_confs, 3)``

The atom/bond feature **column order**, per-column bucket sizes, and cumulative
feature offsets give a fixed feature dimensionality (atom vocab = 62, 16
columns; bond vocab = 16, 5 columns) that the encoder architecture
(``num_atom_features=62`` / ``num_edge_features=16``) expects.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from rdkit import Chem

from ..graph import MolecularGraph

# Categorical value buckets and their fallback indices.
_ATOMIC_NUMBERS = [1, 6, 7, 8, 9, 14, 15, 16, 17, 26, 35, 53]
_ATOMIC_NUMBER_OTHER = 12
_DEGREES = [0, 1, 2, 3, 4]
_DEGREE_OTHER = 5
_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
_FORMAL_CHARGE_NEG = 5
_FORMAL_CHARGE_POS = 6
_IMPLICIT_HS = [0, 1, 2, 3, 4]
_IMPLICIT_H_OTHER = 5
_BOND_ORDERS = [1, 1.5, 2, 3, 4]
_BOND_ORDER_OTHER = 5

# Ring sizes featurised individually as part of the atom feature schema.
_RING_SIZES = (3, 4, 5, 6, 7, 8, 9, 10)

# Per-column sizes. Atom: 16 columns, Bond: 5.
_ATOM_FEATURE_SIZES = [
    len(_ATOMIC_NUMBERS) + 1,  # AtomNumber           (13)
    len(_DEGREES) + 1,  # AtomDegree                   (6)
    len(_FORMAL_CHARGES) + 2,  # AtomFormalCharge      (7)
    len(_IMPLICIT_HS) + 1,  # AtomImplicitHydrogenCount(6)
    2,  # AtomIsAromatic                               (2)
    6,  # AtomHybridization                            (6)
    2,  # AtomIsInRing                                 (2)
    *([2] * len(_RING_SIZES)),  # AtomIsInRingSize(3..10)
    4,  # AtomCIPStereo                                (4)
]
_BOND_FEATURE_SIZES = [
    len(_BOND_ORDERS) + 1,  # BondOrder      (6)
    2,  # BondIsInRing                       (2)
    2,  # BondIsAromatic                     (2)
    2,  # BondIsRotor                        (2)
    4,  # BondCIPStereo                      (4)
]

NUM_ATOM_FEATURES = len(_ATOM_FEATURE_SIZES)  # 16 columns
NUM_BOND_FEATURES = len(_BOND_FEATURE_SIZES)  # 5 columns
ATOM_FEATURE_VOCAB = sum(_ATOM_FEATURE_SIZES)  # 62
BOND_FEATURE_VOCAB = sum(_BOND_FEATURE_SIZES)  # 16


def _offsets(sizes: list[int]) -> np.ndarray:
    """Compute cumulative feature offsets from a list of per-column bucket sizes."""
    offs = [0]
    for size in sizes[:-1]:
        offs.append(offs[-1] + size)
    return np.array(offs)


# Hybridisation: map RDKit HybridizationType to the feature bucket indices
# (0=unknown/other, 1=sp, 2=sp2, 3=sp3, 4=sp3d, 5=sp3d2).
_HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}


def _atom_number_index(atom: Chem.rdchem.Atom) -> int:
    """Return the feature bucket index for an atom's atomic number."""
    num = atom.GetAtomicNum()
    return _ATOMIC_NUMBERS.index(num) if num in _ATOMIC_NUMBERS else _ATOMIC_NUMBER_OTHER


def _atom_degree_index(atom: Chem.rdchem.Atom) -> int:
    """Return the feature bucket index for an atom's connectivity degree."""
    deg = atom.GetDegree()
    return _DEGREES.index(deg) if deg in _DEGREES else _DEGREE_OTHER


def _atom_formal_charge_index(atom: Chem.rdchem.Atom) -> int:
    """Return the feature bucket index for an atom's formal charge."""
    charge = atom.GetFormalCharge()
    if charge in _FORMAL_CHARGES:
        return _FORMAL_CHARGES.index(charge)
    return _FORMAL_CHARGE_NEG if charge < 0 else _FORMAL_CHARGE_POS


def _atom_implicit_h_index(atom: Chem.rdchem.Atom) -> int:
    """Return the feature bucket index for an atom's implicit hydrogen count."""
    # Use total hydrogens for graphs parsed from SMILES, which contain no
    # explicit hydrogen atoms.
    count = atom.GetTotalNumHs()
    return _IMPLICIT_HS.index(count) if count in _IMPLICIT_HS else _IMPLICIT_H_OTHER


def _atom_hybridization_index(atom: Chem.rdchem.Atom) -> int:
    """Return the feature bucket index for an atom's hybridization state."""
    return _HYBRIDIZATION_MAP.get(atom.GetHybridization(), 0)


def _atom_cip_stereo_index(atom: Chem.rdchem.Atom) -> int:
    """Return the feature bucket index for an atom's CIP stereo label."""
    # Feature buckets: 0=not stereo, 1=S, 2=R, 3=unspecified stereocentre.
    if atom.HasProp("_CIPCode"):
        code = atom.GetProp("_CIPCode")
        if code == "S":
            return 1
        if code == "R":
            return 2
        return 3
    if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        return 3
    return 0


def _atom_feature_row(atom: Chem.rdchem.Atom) -> list[int]:
    """Build the full atom feature row as a list of bucket indices."""
    row = [
        _atom_number_index(atom),
        _atom_degree_index(atom),
        _atom_formal_charge_index(atom),
        _atom_implicit_h_index(atom),
        int(atom.GetIsAromatic()),
        _atom_hybridization_index(atom),
        int(atom.IsInRing()),
    ]
    row.extend(int(atom.IsInRingSize(size)) for size in _RING_SIZES)
    row.append(_atom_cip_stereo_index(atom))
    return row


def _bond_order_index(bond: Chem.rdchem.Bond) -> int:
    """Return the feature bucket index for a bond's order."""
    if bond.GetIsAromatic():
        order: float = 1.5
    else:
        order = bond.GetBondTypeAsDouble()
        if order != 1.5 and order == int(order):
            order = int(order)
    return _BOND_ORDERS.index(order) if order in _BOND_ORDERS else _BOND_ORDER_OTHER


def _bond_is_rotor_index(bond: Chem.rdchem.Bond) -> int:
    """Return 1 if the bond is rotatable (single, acyclic, between two non-terminal atoms), else 0."""
    # A rotatable bond is single, acyclic, and joins two non-terminal heavy atoms.
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE or bond.IsInRing():
        return 0
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    return int(begin.GetDegree() > 1 and end.GetDegree() > 1)


def _bond_cip_stereo_index(bond: Chem.rdchem.Bond) -> int:
    """Return the feature bucket index for a bond's CIP stereo label."""
    # Feature buckets: 0=not stereo, 1=E, 2=Z, 3=unspecified.
    stereo = bond.GetStereo()
    if stereo in (Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOTRANS):
        return 1
    if stereo in (Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOCIS):
        return 2
    if stereo == Chem.rdchem.BondStereo.STEREOANY:
        return 3
    return 0


def _bond_feature_row(bond: Chem.rdchem.Bond) -> list[int]:
    """Build the full bond feature row as a list of bucket indices."""
    return [
        _bond_order_index(bond),
        int(bond.IsInRing()),
        int(bond.GetIsAromatic()),
        _bond_is_rotor_index(bond),
        _bond_cip_stereo_index(bond),
    ]


class RDKitGraphFeaturizer:
    """Featurise an RDKit molecule into a :class:`MolecularGraph`."""

    def __init__(
        self,
        permute_edge_index: bool = True,
        as_torch_tensor: bool = False,
        add_feature_offsets: bool = True,
        int_type=np.uint8,
        add_pos: bool = False,
    ) -> None:
        """Args:
            permute_edge_index: Return ``edge_index`` as ``(2, E)`` rather than
                ``(E, 2)``. Defaults to True.
            as_torch_tensor: Convert the resulting graph to torch tensors.
                Defaults to False.
            add_feature_offsets: Add cumulative bucket offsets to feature
                indices, producing globally unique one-hot indices for embedding
                lookup. Defaults to True.
            int_type: NumPy integer dtype for feature arrays. Defaults to
                ``np.uint8``.
            add_pos: Extract and include per-conformer positions as ``pos``.
                Defaults to False.
        """
        self.permute_edge_index = permute_edge_index
        self.as_torch_tensor = as_torch_tensor
        self.add_feature_offsets = add_feature_offsets
        self.int_type = int_type
        self.add_pos = add_pos
        self.atom_feature_offsets = _offsets(_ATOM_FEATURE_SIZES)
        self.bond_feature_offsets = _offsets(_BOND_FEATURE_SIZES)

    def prepare_mol(self, mol: Chem.Mol) -> None:
        """Prepare an RDKit molecule for featurization by assigning stereo info.

        Args:
            mol: RDKit molecule to prepare (modified in-place).
        """
        # Ensure ring info + CIP stereo are perceived. Molecules parsed via
        # Chem.MolFromSmiles are already sanitised; assigning stereochemistry
        # populates the ``_CIPCode`` atom property used above.
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    def get_node_features(self, mol: Chem.Mol) -> np.ndarray:
        """Compute the atom feature matrix for ``mol``.

        Args:
            mol: Prepared RDKit molecule.

        Returns:
            Node feature array of shape ``(num_atoms, 16)``.
        """
        x = np.array([_atom_feature_row(atom) for atom in mol.GetAtoms()])
        if self.add_feature_offsets:
            x = x + self.atom_feature_offsets
        assert x.max() < np.iinfo(self.int_type).max
        return x.astype(self.int_type)

    def get_edge_features(self, mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
        """Compute edge connectivity and bond feature matrix for ``mol``.

        Args:
            mol: Prepared RDKit molecule.

        Returns:
            Tuple of ``(edge_index, edge_attr)`` with shapes ``(2, num_edges)``
            and ``(num_edges, 5)``.
        """
        num_atoms = mol.GetNumAtoms()
        edge_index_list: list[list[int]] = []
        edge_attr_list: list[list[int]] = []
        for bond in mol.GetBonds():
            ba = bond.GetBeginAtomIdx()
            ea = bond.GetEndAtomIdx()
            edge_index_list.extend([[ba, ea], [ea, ba]])
            features = _bond_feature_row(bond)
            edge_attr_list.extend([features, features])

        if not edge_index_list:
            # No bonds (e.g. single-atom molecule): emit empty edge tensors with
            # the correct second dimension so downstream batching still works.
            edge_index = np.zeros((2, 0), dtype=self.int_type)
            edge_attr = np.zeros((0, NUM_BOND_FEATURES), dtype=self.int_type)
            return edge_index, edge_attr

        edge_index = np.array(edge_index_list)
        edge_attr = np.array(edge_attr_list)
        # Sort directed edges deterministically by source then destination.
        perm = (edge_index[:, 0] * num_atoms + edge_index[:, 1]).argsort()
        edge_index, edge_attr = edge_index[perm], edge_attr[perm]
        if self.permute_edge_index:
            edge_index = edge_index.T
        if self.add_feature_offsets:
            edge_attr = edge_attr + self.bond_feature_offsets
        return edge_index.astype(self.int_type), edge_attr.astype(self.int_type)

    def maybe_get_pos(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """Extract per-conformer atom positions if available.

        Args:
            mol: RDKit molecule, possibly with embedded conformers.

        Returns:
            Float array of shape ``(num_atoms, num_confs, 3)``, or ``None`` if
            ``add_pos`` is False, the molecule has no conformers, or all
            positions are zero.
        """
        if not self.add_pos or mol.GetNumConformers() == 0:
            return None
        num_atoms = mol.GetNumAtoms()
        num_confs = mol.GetNumConformers()
        pos = np.zeros((num_atoms, num_confs, 3))
        for i, conf in enumerate(mol.GetConformers()):
            pos[:, i] = conf.GetPositions()
        if np.all(pos == 0):
            return None
        return pos

    def __call__(self, mol: Chem.Mol) -> MolecularGraph:
        """Featurize ``mol`` into a :class:`MolecularGraph`.

        Args:
            mol: RDKit molecule to featurize.

        Returns:
            :class:`MolecularGraph` with node features, edge connectivity, and
            optionally atom positions.
        """
        self.prepare_mol(mol)
        assert mol.GetNumAtoms() < np.iinfo(self.int_type).max, (
            "Molecule has more atoms than can be represented by int type "
            f"{self.int_type}"
        )
        node_features = self.get_node_features(mol)
        edge_index, edge_attr = self.get_edge_features(mol)
        pos = self.maybe_get_pos(mol)
        graph = MolecularGraph(
            x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=pos
        )
        if self.as_torch_tensor:
            graph.to_torch()
        return graph

    def smiles_to_graph(self, smiles: str) -> MolecularGraph:
        """Featurize a SMILES string directly into a :class:`MolecularGraph`.

        Args:
            smiles: SMILES string.

        Returns:
            :class:`MolecularGraph` with node features, edge connectivity, and
            optionally atom positions.

        Raises:
            ValueError: If the SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return self(mol)
