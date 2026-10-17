"""Scored molecular pairs and batching for SAND training."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .mol_utils.graph import MolecularGraph


class SANDMolecularGraphDataset(Dataset):
    """Molecular graph pairs and their similarity scores."""

    def __init__(self, molecular_graph_collection, query_idxs, match_idxs, scores):
        self.molecular_graph_collection = molecular_graph_collection
        self.query_idxs = query_idxs
        self.match_idxs = match_idxs
        self.scores = scores

    def __len__(self):
        return len(self.query_idxs)

    def __getitem__(self, idx):
        query_idx = self.query_idxs[idx]
        match_idx = self.match_idxs[idx]
        score = torch.tensor(self.scores[idx]).float()
        return {
            "query_graph": self.molecular_graph_collection[query_idx],
            "match_graph": self.molecular_graph_collection[match_idx],
            "score": score,
        }

    def collate_fn(self, data_list):
        graphs = []
        for data in data_list:
            graphs.append(data["query_graph"])
            graphs.append(data["match_graph"])
        batch = MolecularGraph.from_graph_list(graphs)
        batch["num_graphs"] = batch["batch_size"]
        batch["score"] = torch.stack([data["score"] for data in data_list], dim=0)
        batch["batch_size"] = len(data_list)
        return batch


class PairBatchSampler:
    """Shuffle pairs while keeping the rank objective defined."""

    def __init__(self, scores, batch_size):
        self.scores = scores
        self.batch_size = batch_size

    def __len__(self):
        count, remainder = divmod(len(self.scores), self.batch_size)
        return max(1, count + (remainder > 1))

    def __iter__(self):
        indices = torch.randperm(len(self.scores)).tolist()
        batches = [
            indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)
        ]
        # Rank loss needs varied target scores; merging or repeating pairs can
        # make a batch larger than batch_size.
        if len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        low, high = int(np.argmin(self.scores)), int(np.argmax(self.scores))
        for batch in batches:
            if np.all(self.scores[batch] == self.scores[batch[0]]):
                batch.append(high if self.scores[batch[0]] == self.scores[low] else low)
            yield batch
