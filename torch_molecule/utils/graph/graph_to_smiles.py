from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import re
import random
import logging
from rdkit import Chem
from typing import List, Tuple, Optional
random.seed(0)
import torch

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

def graph_to_smiles(molecule_list: List[Tuple], atom_decoder: list) -> List[Optional[str]]:
    smiles_list = []
    for index, graph in enumerate(molecule_list):
        try:
            atom_types, edge_types = graph
            mol_init = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
            
            # Try to correct the molecule with connection=True, then False if needed
            for connection in (True, False):
                mol_conn, _ = correct_mol(mol_init, connection=connection)
                if mol_conn is not None:
                    break
            else:
                mol_conn = mol_init  # Fallback to initial molecule

            # Convert to SMILES
            smiles = mol2smiles(mol_conn)
            if not smiles:
                smiles = Chem.MolToSmiles(mol_conn)

            if smiles:
                mol = get_mol(smiles)
                if mol is not None:
                    # Get the largest fragment
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                    largest_mol = max(mol_frags, key=lambda m: m.GetNumAtoms())
                    
                    largest_smiles = mol2smiles(largest_mol)
                    if largest_smiles and len(largest_smiles) > 1:
                        smiles_list.append(largest_smiles)
                    else:
                        smiles_list.append(None)
                else:
                    smiles_list.append(None)
            else:
                smiles_list.append(None)

        except Exception as e:
            try:
                # Fallback to RDKit's MolToSmiles if everything else fails
                fallback_smiles = Chem.MolToSmiles(mol_init)
                if fallback_smiles:
                    smiles_list.append(fallback_smiles)
                else:
                    smiles_list.append(None)
            except Exception as e2:
                smiles_list.append(None)

    return smiles_list

def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(
                bond[0].item(),
                bond[1].item(),
                bond_dict[edge_types[bond[0], bond[1]].item()],
            )
            if verbose:
                print(
                    "bond added:",
                    bond[0].item(),
                    bond[1].item(),
                    edge_types[bond[0], bond[1]].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                if len(atomid_valence) == 2:
                    idx = atomid_valence[0]
                    v = atomid_valence[1]
                    an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                    if verbose:
                        print("atomic num of atom with a large valence", an)
                    if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                        mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                        # print("Formal charge added")
                else:
                    continue
    return mol


def correct_mol(mol, connection=False):
    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        if connection:
            mol_conn = connect_fragments(mol)
            mol = mol_conn
            if mol is None:
                return None, no_correct
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            try:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                queue = []
                check_idx = 0
                for b in mol.GetAtomWithIdx(idx).GetBonds():
                    type = int(b.GetBondType())
                    queue.append(
                        (b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                    )
                    if type == 12:
                        check_idx += 1
                queue.sort(key=lambda tup: tup[1], reverse=True)

                if queue[-1][1] == 12:
                    return None, no_correct
                elif len(queue) > 0:
                    start = queue[check_idx][2]
                    end = queue[check_idx][3]
                    t = queue[check_idx][1] - 1
                    mol.RemoveBond(start, end)
                    if t >= 1:
                        mol.AddBond(start, end, bond_dict[t])
            except Exception as e:
                # print(f"An error occurred in correction: {e}")
                return None, no_correct
    return mol, no_correct


def check_valid(smiles):
    mol = get_mol(smiles)
    if mol is None:
        return False
    smiles = mol2smiles(mol)
    if smiles is None:
        return False
    return True


def get_mol(smiles_or_mol):
    """
    Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def mol2smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def check_valency(mol):
    try:
        # First attempt to sanitize with specific properties
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence
    except Exception as e:
        # print(f"An unexpected error occurred: {e}")
        return False, []


##### connect fragements
def select_atom_with_available_valency(frag):
    atoms = list(frag.GetAtoms())
    random.shuffle(atoms)
    for atom in atoms:
        if atom.GetAtomicNum() > 1 and atom.GetImplicitValence() > 0:
            return atom
    return None


def select_atoms_with_available_valency(frag):
    return [
        atom
        for atom in frag.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetImplicitValence() > 0
    ]


def try_to_connect_fragments(combined_mol, frag, atom1, atom2):
    # Make copies of the molecules to try the connection
    trial_combined_mol = Chem.RWMol(combined_mol)
    trial_frag = Chem.RWMol(frag)

    # Add the new fragment to the combined molecule with new indices
    new_indices = {
        atom.GetIdx(): trial_combined_mol.AddAtom(atom)
        for atom in trial_frag.GetAtoms()
    }

    # Add the bond between the suitable atoms from each fragment
    trial_combined_mol.AddBond(
        atom1.GetIdx(), new_indices[atom2.GetIdx()], Chem.BondType.SINGLE
    )

    # Adjust the hydrogen count of the connected atoms
    for atom_idx in [atom1.GetIdx(), new_indices[atom2.GetIdx()]]:
        atom = trial_combined_mol.GetAtomWithIdx(atom_idx)
        num_h = atom.GetTotalNumHs()
        atom.SetNumExplicitHs(max(0, num_h - 1))

    # Add bonds for the new fragment
    for bond in trial_frag.GetBonds():
        trial_combined_mol.AddBond(
            new_indices[bond.GetBeginAtomIdx()],
            new_indices[bond.GetEndAtomIdx()],
            bond.GetBondType(),
        )

    # Convert to a Mol object and try to sanitize it
    new_mol = Chem.Mol(trial_combined_mol)
    try:
        Chem.SanitizeMol(new_mol)
        return new_mol  # Return the new valid molecule
    except Chem.MolSanitizeException:
        return None  # If the molecule is not valid, return None


def connect_fragments(mol):
    # Get the separate fragments
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) < 2:
        return mol

    combined_mol = Chem.RWMol(frags[0])

    for frag in frags[1:]:
        # Select all atoms with available valency from both molecules
        atoms1 = select_atoms_with_available_valency(combined_mol)
        atoms2 = select_atoms_with_available_valency(frag)

        # Try to connect using all combinations of available valency atoms
        for atom1 in atoms1:
            for atom2 in atoms2:
                new_mol = try_to_connect_fragments(combined_mol, frag, atom1, atom2)
                if new_mol is not None:
                    # If a valid connection is made, update the combined molecule and break
                    combined_mol = new_mol
                    break
            else:
                # Continue if the inner loop didn't break (no valid connection found for atom1)
                continue
            # Break if the inner loop did break (valid connection found)
            break
        else:
            # If no valid connections could be made with any of the atoms, return None
            return None

    return combined_mol