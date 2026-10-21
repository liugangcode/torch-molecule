import re
from rdkit import Chem

ptable = Chem.GetPeriodicTable()
ELEMENTS = [ptable.GetElementSymbol(i) for i in range(0, 119)]  # 0 -> * for polymerization point
ATOM_VALENCY = {Z: Chem.GetPeriodicTable().GetDefaultValence(Z) for Z in range(1, 119)}

from rdkit.Chem.rdchem import BondType

bond_type_map = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
}


def smiles_to_molecule(smiles, kekulize=False, sanitize=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        try:
            atom = Chem.Atom(smiles)
            mol = Chem.RWMol()
            mol.AddAtom(atom)
            mol = mol.GetMol()
        except:
            raise ValueError(f"Invalid SMILES string: {smiles}")
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def molecule_to_smiles(mol, canonical=True, kekulize=False):
    if mol.GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        return atom.GetSymbol()
    return Chem.MolToSmiles(mol, canonical=canonical, kekuleSmiles=kekulize)


def canonicalize(mol_input, kekulize=False, remove_isolated_h=True):
    input_is_smiles = isinstance(mol_input, str)
    if input_is_smiles:
        if input_is_smiles in ELEMENTS:
            return mol_input
        mol = smiles_to_molecule(mol_input, kekulize=kekulize)
        # if input is an single atom like Sn directly return it
    else:
        if not isinstance(mol_input, Chem.rdchem.Mol):
            raise ValueError("Input must be either a SMILES string or RDKit molecule")
        mol = mol_input

    canonical_smiles = molecule_to_smiles(mol)
    if input_is_smiles:
        if kekulize and mol.GetNumAtoms() > 1:
            mol = smiles_to_molecule(canonical_smiles)
            canonical_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
        return canonical_smiles
    else:
        canonical_mol = smiles_to_molecule(canonical_smiles, kekulize=kekulize)
        return canonical_mol


def get_indexed_smiles(mol_input, kekulize=False):
    if isinstance(mol_input, str):
        mol = smiles_to_molecule(mol_input, kekulize)
    else:
        if not isinstance(mol_input, Chem.rdchem.Mol):
            raise ValueError("Input must be either a SMILES string or RDKit molecule")
        mol = mol_input
        if kekulize:
            Chem.Kekulize(mol, True)

    # Add atom indices as mapping numbers
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("molAtomMapNumber", str(i))

    if mol.GetNumAtoms() == 1:
        atom = mol.GetAtomWithIdx(0)
        symbol = atom.GetSymbol()
        return f"[{symbol}:0]"
    else:
        indexed_smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=kekulize)
        return indexed_smiles


def get_sub_molecule(mol, atom_indices, kekulize=False):
    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        return smiles_to_molecule(atom_symbol, kekulize)
    atom_index_dict = {i: True for i in atom_indices}
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        if begin_atom_idx in atom_index_dict and end_atom_idx in atom_index_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol


def count_atom(smiles, return_dict=False):
    # Initialize dictionary to count atoms in ELEMENTS
    atom_dict = {atom: 0 for atom in ELEMENTS}
    i = 0
    while i < len(smiles):
        # Get the current character (potential atom symbol)
        symbol = smiles[i].upper()
        next_char = smiles[i + 1] if i + 1 < len(smiles) else None

        if next_char:
            combined = symbol + next_char
            # Check if either combination exists in atom_dict
            if combined in atom_dict or f"[{combined}]" in atom_dict:
                # Determine which form exists and should be used
                if f"[{combined}]" in atom_dict:
                    symbol = combined
                elif combined in atom_dict:
                    symbol = combined
                i += 1

        # Check if the current symbol exists in atom_dict (either directly or bracketed)
        if symbol in atom_dict or f"[{symbol}]" in atom_dict:
            # If both forms exist, prefer the bracketed version
            key = f"[{symbol}]" if f"[{symbol}]" in atom_dict else symbol
            atom_dict[key] += 1
        i += 1

    # Return results as dictionary or total count
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())


def check_valency(mol):
    """
    Check if any atom in the molecule violates expected valency.
    Returns:
    - True, None if valency is correct.
    - False, list of problematic atom indices if valency is violated.
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_charge(mol_edit):
    """
    Corrects formal charges on atoms with incorrect valency.
    Returns:
    - Corrected RDKit molecule object.
    """
    is_valid, atomid_valence = check_valency(mol_edit)
    if is_valid:
        return mol_edit  # No correction needed

    idx = 0
    while len(atomid_valence) == 2:
        atomid, valency = atomid_valence
        atom = mol_edit.GetAtomWithIdx(atomid)
        atomic_num = atom.GetAtomicNum()
        # Only correct valency if the element is in our list
        if atomic_num in ATOM_VALENCY:
            expected_valency = ATOM_VALENCY[atomic_num]
            valency_diff = valency - expected_valency

            if valency_diff == 1 and atomic_num in [6, 7, 8, 16]:  # C, N, O, S
                atom.SetFormalCharge(1)
            elif valency_diff == 1 and atomic_num == 5:  # B
                atom.SetFormalCharge(-1)
            else:
                break

        is_valid, atomid_valence = check_valency(mol_edit)
        if is_valid:
            break
        idx += 1
        if idx > 10:
            break

    return mol_edit
