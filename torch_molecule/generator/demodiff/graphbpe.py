import os
import json
import random
from copy import copy

from .token_utils import (
    canonicalize,
    get_indexed_smiles,
    get_sub_molecule,
    count_atom,
    smiles_to_molecule,
    molecule_to_smiles,
    ELEMENTS,
    bond_type_map,
    correct_charge,
)

# used in decode
from rdkit import Chem
from rdkit.Chem import RWMol


class MolecularSubgraphProcessor:
    def __init__(self, smiles, molecule, kekulize=False, simple_mode=False, allowable_rings=None):
        self.smiles = smiles
        self.molecule = molecule
        self.kekulize = kekulize
        self.simple_mode = simple_mode

        # Get rings information
        self.rings = self.find_rings()

        self.subgraphs = {}
        self.subgraphs_smiles = {}
        self.upid_cnt = 0

        # Initialize rings
        ring_atoms = set()
        self.no_merge_atoms = set()
        if not simple_mode:
            for ring in self.rings:
                ring_submol = get_sub_molecule(molecule, list(ring), kekulize)
                ring_smiles = molecule_to_smiles(ring_submol, kekulize)

                if allowable_rings is not None and ring_smiles not in allowable_rings:
                    self.no_merge_atoms.update(ring)
                    for idx in ring:
                        self.subgraphs[self.upid_cnt] = {
                            idx: molecule.GetAtomWithIdx(idx).GetSymbol()
                        }
                        self.subgraphs_smiles[self.upid_cnt] = molecule.GetAtomWithIdx(
                            idx
                        ).GetSymbol()
                        self.upid_cnt += 1
                else:
                    ring_submol = get_sub_molecule(molecule, list(ring), kekulize)
                    ring_smiles = molecule_to_smiles(ring_submol, kekulize)
                    # ring_smiles = canonicalize(ring_smiles, kekulize)
                    self.subgraphs[self.upid_cnt] = {
                        idx: molecule.GetAtomWithIdx(idx).GetSymbol() for idx in ring
                    }
                    self.subgraphs_smiles[self.upid_cnt] = ring_smiles
                    ring_atoms.update(ring)
                    self.upid_cnt += 1

        # Initialize remaining non-ring atoms
        for atom in molecule.GetAtoms():
            idx = atom.GetIdx()
            if idx not in ring_atoms and idx not in self.no_merge_atoms:
                self.subgraphs[self.upid_cnt] = {idx: atom.GetSymbol()}
                self.subgraphs_smiles[self.upid_cnt] = atom.GetSymbol()
                self.upid_cnt += 1

        # Map atoms to subgraph IDs
        self.inversed_index = {}
        for aid in range(molecule.GetNumAtoms()):
            for key in self.subgraphs:
                if aid in self.subgraphs[key]:
                    self.inversed_index[aid] = key

        self.dirty = True
        self.smi2pids = {}

    def find_rings(self):
        """Find largest connected ring systems in the molecule, including spiro and bridged rings."""
        ring_info = self.molecule.GetRingInfo()
        bond_rings = ring_info.BondRings()
        atom_rings = ring_info.AtomRings()

        # Map atoms to their rings
        atom_to_rings = {}
        for i, ring in enumerate(atom_rings):
            for atom in ring:
                if atom not in atom_to_rings:
                    atom_to_rings[atom] = set()
                atom_to_rings[atom].add(i)

        connected_systems = []
        used_rings = set()

        for i, ring in enumerate(bond_rings):
            if i in used_rings:
                continue

            current_system = set(ring)
            ring_indices = {i}
            atoms_in_system = set(atom_rings[i])

            changed = True
            while changed:
                changed = False
                connected_atoms = set()
                for atom in atoms_in_system:
                    connected_atoms.update(atom_to_rings[atom])

                for ring_idx in connected_atoms:
                    if ring_idx not in ring_indices:
                        current_system.update(bond_rings[ring_idx])
                        atoms_in_system.update(atom_rings[ring_idx])
                        ring_indices.add(ring_idx)
                        used_rings.add(ring_idx)
                        changed = True

            if ring_indices:
                # Convert bond indices to atom indices
                atom_system = set()
                for bond_idx in current_system:
                    bond = self.molecule.GetBondWithIdx(bond_idx)
                    atom_system.add(bond.GetBeginAtomIdx())
                    atom_system.add(bond.GetEndAtomIdx())
                connected_systems.append(atom_system)

        return connected_systems

    def get_nei_subgraphs(self):
        nei_subgraphs, merge_pids = [], []
        for key in self.subgraphs:
            subgraph = self.subgraphs[key]
            local_nei_pid = []
            for aid in subgraph:
                atom = self.molecule.GetAtomWithIdx(aid)
                if aid in self.no_merge_atoms:
                    continue
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if (
                        nei_idx in subgraph
                        or nei_idx > aid
                        or nei_idx in self.no_merge_atoms
                        # nei_idx in subgraph or nei_idx > aid
                    ):  # don't consider (1) within the subgraph, (2) connecting to atoms with larger index, (3) connecting to no-merge atoms
                        continue
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_subgraph = copy(subgraph)
                new_subgraph.update(self.subgraphs[nei_pid])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((key, nei_pid))
        return nei_subgraphs, merge_pids

    def get_nei_smis(self):
        if self.dirty:
            nei_subgraphs, merge_pids = self.get_nei_subgraphs()
            nei_smis, self.smi2pids = [], {}
            for i, subgraph in enumerate(nei_subgraphs):
                submol = get_sub_molecule(
                    self.molecule, list(subgraph.keys()), kekulize=self.kekulize
                )
                smiles = molecule_to_smiles(submol, self.kekulize)
                nei_smis.append(smiles)
                self.smi2pids.setdefault(smiles, [])
                self.smi2pids[smiles].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis

    def merge(self, smi):
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids:
                if pid1 in self.subgraphs and pid2 in self.subgraphs:
                    # if any(aid in self.no_merge_atoms for aid in self.subgraphs[pid1]) or any(aid in self.no_merge_atoms for aid in self.subgraphs[pid2]):
                    #     continue
                    self.subgraphs[pid1].update(self.subgraphs[pid2])
                    self.subgraphs[self.upid_cnt] = self.subgraphs[pid1]
                    self.subgraphs_smiles[self.upid_cnt] = smi
                    for aid in self.subgraphs[pid2]:
                        self.inversed_index[aid] = pid1
                    for aid in self.subgraphs[pid1]:
                        self.inversed_index[aid] = self.upid_cnt
                    del self.subgraphs[pid1]
                    del self.subgraphs[pid2]
                    del self.subgraphs_smiles[pid1]
                    del self.subgraphs_smiles[pid2]
                    self.upid_cnt += 1
        self.dirty = True

    def get_smis_subgraphs(self):
        results = []
        for pid in self.subgraphs_smiles:
            smiles = self.subgraphs_smiles[pid]
            group_dict = self.subgraphs[pid]
            indices = list(group_dict.keys())

            if len(indices) == 1:
                # Handle single-atom case
                new_to_old_dict = {0: indices[0]}
                results.append((smiles, indices, new_to_old_dict))
                continue

            mol_copy = Chem.RWMol(self.molecule)
            for orig_idx in indices:
                atom = mol_copy.GetAtomWithIdx(orig_idx)
                atom.SetProp("molOriginalIdx", str(orig_idx))  # Store original index

            mol = get_sub_molecule(mol_copy, indices, kekulize=self.kekulize)
            smi = Chem.MolToSmiles(mol)
            order = mol.GetPropsAsDict(True, True)["_smilesAtomOutputOrder"]
            mol_canonical = Chem.RenumberAtoms(mol, order)

            new_to_old_dict = {}
            for atom in mol_canonical.GetAtoms():  # does not follow the canonical order
                new_to_old_dict[atom.GetIdx()] = int(atom.GetProp("molOriginalIdx"))
            results.append((smiles, indices, new_to_old_dict))
        return results


class MolecularGraphTokenizer:
    def __init__(self, kekulize, name, simple_mode=False):
        self.kekulize = kekulize
        self.vocab_node = []
        self.vocab_node_indexed = []
        self.vocab_node_stats = {}
        self.vocab_edge = []
        self.vocab_edge_stats = {}
        self.initial_rings = []
        self.max_atom_in_token = None
        self.max_node_type = None
        self.max_edge_type = None
        self.max_bond_type = len(bond_type_map)
        self.unknown = []
        self.simple_mode = simple_mode
        self.name = name

    def load(self, model_file):
        with open(f"{model_file}.node", "r") as fin:
            lines = fin.read().strip().split("\n")
        config = json.loads(lines[0])
        self.kekulize = config["kekulize"]
        lines = lines[1:]

        self.vocab_node_stats = {}
        self.vocab_node = []
        self.vocab_node_indexed = []
        self.max_atom_in_token = 0
        for line in lines:
            smiles, atom_num, frequency, indexed_smiles = line.strip().split("\t")
            self.vocab_node_stats[smiles] = (int(atom_num), int(frequency))
            self.max_atom_in_token = max(self.max_atom_in_token, int(atom_num))
            self.vocab_node.append(smiles)
            self.vocab_node_indexed.append(indexed_smiles)
        self.max_node_type = len(self.vocab_node)

        if os.path.exists(f"{model_file}.edge"):
            with open(f"{model_file}.edge", "r") as fin:
                lines = fin.read().strip().split("\n")
            vocab_edge = []
            vocab_edge_stats = {}
            for line in lines:
                # Split line into edge_type string and statistics
                edge_part, num_attrs, frequency = line.strip().split("\t")
                # Convert edge_type string to tuple
                edge_type = tuple(map(int, edge_part.strip("() ").split(", ")))
                vocab_edge.append(edge_type)
                vocab_edge_stats[edge_type] = (int(num_attrs), int(frequency))
            self.vocab_edge = vocab_edge
            self.vocab_edge_stats = vocab_edge_stats
            self.max_edge_type = len(vocab_edge)
        else:
            print("Edge vocabulary is not initialized")
            self.vocab_edge = []
            self.vocab_edge_stats = []
            self.max_edge_type = None

        if os.path.exists(f"{model_file}.ring"):
            with open(f"{model_file}.ring", "r") as fin:
                self.initial_rings = fin.read().strip().split("\n")

    def encode(self, input_smiles, update_vocab_edge=True):
        input_smiles_cano = canonicalize(input_smiles, self.kekulize)
        original_molecule = smiles_to_molecule(input_smiles_cano, kekulize=self.kekulize)
        molecule = MolecularSubgraphProcessor(
            input_smiles_cano,
            original_molecule,
            kekulize=self.kekulize,
            simple_mode=self.simple_mode,
            allowable_rings=self.initial_rings,
        )

        idx = 0
        while True:
            idx += 1
            # each iteration extend one-hop neighnbor
            neighbor_smiles = molecule.get_nei_smis()
            max_frequency, smiles_for_merge = -1, ""

            # for k-hop neighor subgraphs, only merge most frequent subgraph (neighborhood) into a single node
            for smiles in neighbor_smiles:
                ## search all subgraphs (neighborhood) to indentify the most frequent one
                if smiles not in self.vocab_node:
                    continue
                frequency = self.vocab_node_stats[smiles][1]
                if frequency > max_frequency:
                    max_frequency, smiles_for_merge = frequency, smiles

            if max_frequency == -1:
                # if all subgraph not in vocabulary, stop searching longer neighbors
                break

            ## merge the subgraph into a single node
            molecule.merge(smiles_for_merge)

        subgraph_results = molecule.get_smis_subgraphs()
        num_nodes = len(subgraph_results)
        node_types = []

        atom_to_subgraph_pos = {}  # Maps original_atom_idx -> (subgraph_idx, vocab_position)
        for subgraph_idx, (subgraph_smiles, atom_indices, new_to_old_dict) in enumerate(
            subgraph_results
        ):
            if subgraph_smiles in self.vocab_node:
                node_idx = self.vocab_node.index(subgraph_smiles)
                node_types.append(node_idx)
                vocab_smiles = self.vocab_node_indexed[node_idx]
                vocab_mol = smiles_to_molecule(vocab_smiles, self.kekulize)
                # vocab_mol = get_sub_molecule(original_molecule, atom_indices, kekulize=self.kekulize)
            else:
                node_idx = len(self.vocab_node)
                self.unknown.append(subgraph_smiles)
                node_types.append(node_idx)
                vocab_smiles = subgraph_smiles
                # vocab_mol = get_sub_molecule(original_molecule, atom_indices, kekulize=self.kekulize)
                vocab_mol = smiles_to_molecule(vocab_smiles, self.kekulize)
                for i, atom in enumerate(vocab_mol.GetAtoms()):
                    atom.SetProp("molAtomMapNumber", str(i))
                ### note: here the code may cause the position exceeding the maximum_atom_in_token
                ### because some larger (ring) subgraphs are encoded as unknown

            for atom in vocab_mol.GetAtoms():
                atom_to_subgraph_pos[new_to_old_dict[atom.GetIdx()]] = (
                    subgraph_idx,
                    int(atom.GetProp("molAtomMapNumber")),
                )

        edge_existence = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        adj = [[-1 for _ in range(num_nodes)] for _ in range(num_nodes)]

        # Process edges
        for bond in original_molecule.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            subgraph_i, subgraph_j = (
                atom_to_subgraph_pos[begin_idx][0],
                atom_to_subgraph_pos[end_idx][0],
            )
            if subgraph_i != subgraph_j:
                pos_i = atom_to_subgraph_pos[begin_idx][1]
                pos_j = atom_to_subgraph_pos[end_idx][1]
                bond_type = bond_type_map[bond.GetBondType()]
                attr_i = (pos_i, bond_type)
                attr_j = (pos_j, bond_type)

                if edge_existence[subgraph_i][subgraph_j] == 1:
                    prev_attr_i = adj[subgraph_i][subgraph_j]
                    prev_attr_j = adj[subgraph_j][subgraph_i]
                    prev_attr_i = self.vocab_edge[prev_attr_i]
                    prev_attr_j = self.vocab_edge[prev_attr_j]
                    attr_i = prev_attr_i + attr_i
                    attr_j = prev_attr_j + attr_j
                else:
                    edge_existence[subgraph_i][subgraph_j] = edge_existence[subgraph_j][
                        subgraph_i
                    ] = 1

                if update_vocab_edge:
                    if attr_i not in self.vocab_edge:
                        self.vocab_edge.append(attr_i)
                        self.vocab_edge_stats[attr_i] = (len(attr_i) // 2, 1)
                    else:
                        num, freq = self.vocab_edge_stats[attr_i]
                        self.vocab_edge_stats[attr_i] = (num, freq + 1)

                    if attr_j not in self.vocab_edge:
                        self.vocab_edge.append(attr_j)
                        self.vocab_edge_stats[attr_j] = (len(attr_j) // 2, 1)
                    else:
                        num, freq = self.vocab_edge_stats[attr_j]
                        self.vocab_edge_stats[attr_j] = (num, freq + 1)

                if attr_i in self.vocab_edge:
                    adj[subgraph_i][subgraph_j] = self.vocab_edge.index(attr_i)
                if attr_j in self.vocab_edge:
                    adj[subgraph_j][subgraph_i] = self.vocab_edge.index(attr_j)

        self.max_edge_type = len(self.vocab_edge)
        return node_types, adj

    def get_bond_position_by_vocab(self, adj_combined):
        num_subgraphs = len(adj_combined)
        bond_adj = [[-1 for _ in range(num_subgraphs)] for _ in range(num_subgraphs)]
        position_adj = [[-1 for _ in range(num_subgraphs)] for _ in range(num_subgraphs)]
        for i in range(num_subgraphs):
            for j in range(i + 1, num_subgraphs):
                if adj_combined[i][j] >= 0:
                    edge_pos_i, bond_type_i = self.vocab_edge[adj_combined[i][j]]
                    edge_pos_j, bond_type_j = self.vocab_edge[adj_combined[j][i]]
                    bond_adj[i][j] = bond_type_i
                    bond_adj[j][i] = bond_type_j
                    position_adj[i][j] = edge_pos_i
                    position_adj[j][i] = edge_pos_j
        return bond_adj, position_adj

    def decode(self, node_types, bond_adj, position_adj, replace_unknown_with_random=True):
        mol = RWMol()
        # Track mapping between (subgraph_idx, vocab_position) -> new_atom_idx
        position_to_atom_idx = {}
        skip_subgraph = []
        contain_unknown = False

        # First pass: Add all atoms from subgraphs
        for subgraph_idx, vocab_idx in enumerate(node_types):
            if vocab_idx >= len(self.vocab_node_indexed):
                contain_unknown = True
                if self.unknown and len(self.unknown) > 0 and replace_unknown_with_random:
                    subgraph_smiles = random.choice(self.unknown)
                else:
                    skip_subgraph.append(subgraph_idx)
                    continue
            else:
                subgraph_smiles = self.vocab_node_indexed[vocab_idx]

            subgraph_mol = smiles_to_molecule(subgraph_smiles, self.kekulize)
            for atom in subgraph_mol.GetAtoms():
                new_atom = Chem.Atom(atom.GetSymbol())
                # Preserve atom properties
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_atom.SetChiralTag(atom.GetChiralTag())
                if atom.GetIsotope():
                    new_atom.SetIsotope(atom.GetIsotope())
                new_idx = mol.AddAtom(new_atom)

                if atom.HasProp("molAtomMapNumber"):
                    position = int(atom.GetProp("molAtomMapNumber"))
                else:
                    position = atom.GetIdx()
                position_to_atom_idx[(subgraph_idx, position)] = new_idx

            # Add bonds within subgraph
            for bond in subgraph_mol.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                if begin_atom.HasProp("molAtomMapNumber"):
                    begin_pos = int(begin_atom.GetProp("molAtomMapNumber"))
                else:
                    begin_pos = begin_atom.GetIdx()
                if end_atom.HasProp("molAtomMapNumber"):
                    end_pos = int(end_atom.GetProp("molAtomMapNumber"))
                else:
                    end_pos = end_atom.GetIdx()
                begin_idx = position_to_atom_idx[(subgraph_idx, begin_pos)]
                end_idx = position_to_atom_idx[(subgraph_idx, end_pos)]

                bond_type = bond.GetBondType()
                mol.AddBond(begin_idx, end_idx, bond_type)
                new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
                new_bond.SetStereo(bond.GetStereo())
                new_bond.SetBondDir(bond.GetBondDir())

        # Second pass: Add bonds between subgraphs
        # except bond type, this may lose other bond information
        num_subgraphs = len(node_types)
        for i in range(num_subgraphs):
            for j in range(i + 1, num_subgraphs):
                if i in skip_subgraph or j in skip_subgraph:
                    continue

                if bond_adj[i][j] >= 0:
                    if position_adj[i][j] == -1:
                        position_adj[i][j] = 0
                    if position_adj[j][i] == -1:
                        position_adj[j][i] = 0

                    bond_type_idx = bond_adj[i][j]
                    pos_i = position_adj[i][j]
                    pos_j = position_adj[j][i]
                    try:
                        atom_i = position_to_atom_idx[(i, pos_i)]
                        atom_j = position_to_atom_idx[(j, pos_j)]
                    except:
                        # the current subgraph doesn't have such a position
                        continue
                    for bt, idx in bond_type_map.items():
                        if idx == bond_type_idx:
                            bond_type = bt
                            break
                    try:
                        mol.AddBond(atom_i, atom_j, bond_type)
                    except:
                        pass

        # Convert to regular molecule and get SMILES
        mol = correct_charge(mol)
        final_mol = mol.GetMol()

        try:
            final_smiles = molecule_to_smiles(final_mol, canonical=True, kekulize=self.kekulize)
        except:
            return None, None, contain_unknown

        try:
            return canonicalize(final_smiles, kekulize=False), None, contain_unknown
        except:
            return None, final_smiles, contain_unknown
