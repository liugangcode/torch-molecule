from collections import defaultdict
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .graphbpe import MolecularGraphTokenizer


class DemoDiffTokenizer:
    """Convert SMILES to the original DemoDiff fragment graphs and back."""

    def __init__(self):
        self.tokenizer = MolecularGraphTokenizer(kekulize=True, name="pretrain")
        self.tokenizer.load(str(Path(__file__).parent / "assets" / "pretrain-token"))
        self.num_node_types = self.tokenizer.max_node_type + 1
        self.num_edge_types = self.tokenizer.max_bond_type + 1
        self.num_position_types = self.tokenizer.max_atom_in_token + 1

    def encode(self, smiles):
        nodes, adjacency = self.tokenizer.encode(smiles, update_vocab_edge=False)
        bonds, positions = self.tokenizer.get_bond_position_by_vocab(adjacency)
        bonds = torch.tensor(bonds, dtype=torch.long) + 1
        positions = torch.tensor(positions, dtype=torch.long) + 1
        edge_index = bonds.nonzero().t().contiguous()
        edge_positions = positions[edge_index[0], edge_index[1]]
        edge_positions[edge_positions >= self.num_position_types] = 1
        return Data(
            x=torch.tensor(nodes, dtype=torch.long),
            edge_index=edge_index,
            edge_attr=bonds[edge_index[0], edge_index[1]],
            edge_pos=edge_positions,
            y=torch.empty(1, 0),
        )

    def decode(self, node_ids, bond_ids, position_ids):
        smiles, _, _ = self.tokenizer.decode(
            torch.as_tensor(node_ids).tolist(),
            (torch.as_tensor(bond_ids) - 1).tolist(),
            (torch.as_tensor(position_ids) - 1).tolist(),
            replace_unknown_with_random=False,
        )
        return smiles


def compute_dataset_info(graphs):
    """Combine the original vocabulary frequencies with training graph statistics."""
    tokenizer = DemoDiffTokenizer()
    num_nodes = tokenizer.num_node_types
    num_edges = tokenizer.num_edge_types
    node_frequency = torch.tensor(
        [count for _, count in tokenizer.tokenizer.vocab_node_stats.values()] + [0],
        dtype=torch.float64,
    )
    edge_frequency = torch.zeros(num_edges, dtype=torch.float64)
    position_frequency = torch.zeros(tokenizer.num_position_types, dtype=torch.float64)
    for (position, bond), (_, count) in tokenizer.tokenizer.vocab_edge_stats.items():
        edge_frequency[bond + 1] += count
        position_frequency[position + 1] += count
    max_n_nodes = max(graph.num_nodes for graph in graphs)
    node_counts = torch.zeros(max_n_nodes + 1, dtype=torch.long)
    pair_counts = defaultdict(lambda: torch.zeros(num_edges, dtype=torch.float64))

    for graph in graphs:
        n = graph.num_nodes
        node_counts[n] += 1
        types, counts = torch.unique(graph.x[graph.x < num_nodes - 1], return_counts=True)
        for a, count_a in zip(types.tolist(), counts.tolist()):
            for b, count_b in zip(types.tolist(), counts.tolist()):
                pair_counts[a, b][0] += count_a * (count_b - (a == b))
        for source, target, bond in zip(
            graph.edge_index[0].tolist(),
            graph.edge_index[1].tolist(),
            graph.edge_attr.tolist(),
        ):
            source_type, target_type = int(graph.x[source]), int(graph.x[target])
            if source_type == num_nodes - 1 or target_type == num_nodes - 1:
                continue
            counts = pair_counts[source_type, target_type]
            counts[0] -= 1
            counts[bond] += 1

    null_count = 2 * sum(counts[0] for counts in pair_counts.values())
    edge_frequency[0] = position_frequency[0] = null_count

    # Match the original sparse expansion: unknown connects only to token zero.
    conditions = torch.zeros(num_nodes, num_edges, dtype=torch.float64)
    conditions[:-1, 0] = num_nodes - 1
    conditions[0, 0] += 1
    conditions[-1, 0] = 1
    for (source, _), counts in pair_counts.items():
        if counts.sum() > 0:
            conditions[source] += counts / counts.sum()
            conditions[source, 0] -= 1

    return {
        "max_n_nodes": max_n_nodes,
        "node_counts": node_counts.tolist(),
        "node_types": (node_frequency / node_frequency.sum()).tolist(),
        "edge_types": (edge_frequency / edge_frequency.sum()).tolist(),
        "pos_types": (position_frequency / position_frequency.sum()).tolist(),
        "xe_conditions": (conditions / (conditions.sum(dim=-1, keepdim=True) + 1e-10)).tolist(),
        "ex_conditions": (
            conditions.t() / (conditions.t().sum(dim=-1, keepdim=True) + 1e-10)
        ).tolist(),
    }


class ContextDataset(Dataset):
    def __init__(self, graphs, contexts=None, score=None, drop_incontext=0.0):
        self.graphs = graphs
        self.contexts = contexts
        self.score = score
        self.drop_incontext = drop_incontext

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        sequence = [index]
        score = [1.0]
        if self.contexts is not None:
            context = self.contexts[index]
            values = self.score[index]
            count = len(context)
            for position, (molecule, value) in enumerate(zip(context, values)):
                drop = self.drop_incontext * (count - position) / count
                if self.drop_incontext <= 0 or random.random() > drop:
                    sequence.append(molecule)
                    score.append(value)
        return {
            "row_id": index,
            "target": index,
            "sequence": sequence,
            "score": score,
            "molecular_graphs": [self.graphs[i] for i in sequence],
        }


def context_collate_fn(batch):
    return {
        "target_id": [item["target"] for item in batch],
        "molecular_graphs": Batch.from_data_list(
            [graph for item in batch for graph in item["molecular_graphs"]]
        ),
        "score": torch.tensor(
            [value for item in batch for value in item["score"]],
            dtype=torch.float32,
        ),
        "context_indicators": torch.tensor(
            [i for i, item in enumerate(batch) for _ in item["sequence"]],
            dtype=torch.long,
        ),
    }
