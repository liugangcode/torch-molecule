import pytest
import torch
from rdkit import Chem
from types import SimpleNamespace
from torch_geometric.data import Data

from torch_molecule.generator.demodiff.data import (
    ContextDataset,
    DemoDiffTokenizer,
    compute_dataset_info,
    context_collate_fn,
)


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "*CCOCC*"])
def test_fragment_graph_roundtrip_preserves_molecular_structure(smiles):
    tokenizer = DemoDiffTokenizer()
    graph = tokenizer.encode(smiles)
    bonds = torch.zeros((graph.num_nodes, graph.num_nodes), dtype=torch.long)
    positions = torch.zeros_like(bonds)
    bonds[graph.edge_index[0], graph.edge_index[1]] = graph.edge_attr
    positions[graph.edge_index[0], graph.edge_index[1]] = graph.edge_pos
    decoded = tokenizer.decode(graph.x, bonds, positions)
    original = Chem.MolFromSmiles(smiles)
    restored = Chem.MolFromSmiles(decoded)
    assert restored is not None
    assert Chem.MolToSmiles(restored) == Chem.MolToSmiles(original)


def test_context_batch_preserves_target_order_and_similarity_scores(smiles):
    tokenizer = DemoDiffTokenizer()
    graphs = [tokenizer.encode(value) for value in smiles]
    contexts = [[2, 1], [3], [], [0]]
    score = [[0.1, 0.7], [0.8], [], [0.3]]
    dataset = ContextDataset(graphs, contexts, score)
    first, third = dataset[0], dataset[2]
    assert first["sequence"] == [0, 2, 1]
    assert first["score"] == [1.0, 0.1, 0.7]
    assert third["sequence"] == [2]
    batch = context_collate_fn([first, third])
    assert batch["target_id"] == [0, 2]
    assert batch["context_indicators"].tolist() == [0, 0, 0, 1]
    torch.testing.assert_close(batch["score"], torch.tensor([1.0, 0.1, 0.7, 1.0]))
    recovered = batch["molecular_graphs"].to_data_list()
    for actual, expected in zip(recovered, [graphs[0], graphs[2], graphs[1], graphs[2]]):
        torch.testing.assert_close(actual.x, expected.x)
        torch.testing.assert_close(actual.edge_pos, expected.edge_pos)


def test_single_fragment_dataset_has_finite_diffusion_distributions():
    tokenizer = DemoDiffTokenizer()
    graphs = [tokenizer.encode(value) for value in ["C", "N", "O"]]
    assert all(graph.num_nodes == 1 for graph in graphs)
    info = compute_dataset_info(graphs)
    assert info["node_counts"] == [0, 3]
    for name in ["node_types", "edge_types", "pos_types"]:
        probabilities = torch.tensor(info[name])
        assert torch.isfinite(probabilities).all()
        assert probabilities.sum() == pytest.approx(1.0)
    assert info["edge_types"][0] == 0.0
    assert info["pos_types"][0] == 0.0


def test_original_vocabulary_priors_and_pair_conditionals(monkeypatch):
    tokenizer = SimpleNamespace(
        num_node_types=4,
        num_edge_types=4,
        num_position_types=3,
        tokenizer=SimpleNamespace(
            vocab_node_stats={"A": (1, 2), "B": (1, 3), "C": (1, 5)},
            vocab_edge_stats={(0, 0): (1, 7), (1, 1): (1, 3), (1, 0): (1, 2)},
        ),
    )
    monkeypatch.setattr(
        "torch_molecule.generator.demodiff.data.DemoDiffTokenizer", lambda: tokenizer
    )
    graph = Data(
        x=torch.tensor([0, 1, 1]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor([1, 1, 2, 2]),
        edge_pos=torch.tensor([1, 1, 2, 2]),
    )
    info = compute_dataset_info([graph])
    assert info["node_counts"] == [0, 0, 0, 1]
    assert info["node_types"] == pytest.approx([0.2, 0.3, 0.5, 0.0])
    assert info["edge_types"] == pytest.approx([4 / 16, 9 / 16, 3 / 16, 0])
    assert info["pos_types"] == pytest.approx([4 / 16, 7 / 16, 5 / 16])
    expected = torch.tensor([[3.5, 0.5, 0, 0], [1.5, 0.5, 1, 0], [3, 0, 0, 0], [1, 0, 0, 0]])
    torch.testing.assert_close(
        torch.tensor(info["xe_conditions"]), expected / expected.sum(-1, keepdim=True)
    )
    torch.testing.assert_close(
        torch.tensor(info["ex_conditions"]),
        expected.t() / (expected.t().sum(-1, keepdim=True) + 1e-10),
    )
