import torch
import random
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS

def get_fingerprint_loss(smiles, h_rep):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    fingerprint_list = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)) for x in smiles]
    fingerprint_loss = 0
    
    for i in range(len(h_rep)):
        for j in range(len(h_rep)-i-1):
            finger_sim = DataStructs.FingerprintSimilarity(fingerprint_list[i], fingerprint_list[i+j+1]) #* 10
            emb_sim = (cos(h_rep[i], h_rep[i+j+1]) + 1)/2
            fingerprint_loss += (finger_sim - emb_sim)**2

    return fingerprint_loss

def get_mask_indices(batched_data, mask_rate=0.15):
    masked_node_indices_full = list()
    offset = 0
    
    for i in range(len(batched_data)):
        masked_node_indices = list()
        data = batched_data[i]
        mol = Chem.MolFromSmiles(data.smiles)
        motifs_list, edges = get_motifs_edges(mol)
        random.shuffle(motifs_list)
        num_atoms = mol.GetNumAtoms()
        sample_size = int(num_atoms * mask_rate + 1)
        
        for motif in motifs_list:
            edge_index_mask = sum(data.edge_index[0]==x for x in motif).bool()
            neigh = torch.masked_select(data.edge_index[1], edge_index_mask)

            if any(x in neigh for x in masked_node_indices):
                continue

            if len(masked_node_indices) + len(motif) > sample_size:
                break
            if len(motif) > 9:
                del motif[5::6]
            for atom in motif:
                masked_node_indices.append(atom + offset)
            
        atom_candidates = [x for x in range(0, num_atoms) if x not in masked_node_indices]
        random.shuffle(atom_candidates)
        if len(masked_node_indices) < sample_size:
            for atom in atom_candidates[:sample_size-len(masked_node_indices)]:
                masked_node_indices.append(atom + offset)
                
        offset += num_atoms
            
        masked_node_indices_full += masked_node_indices

    return masked_node_indices_full

def get_motifs_edges(data):
    Chem.SanitizeMol(data)
    motifs, edges = brics_decomp(data)
    return motifs, edges

def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges