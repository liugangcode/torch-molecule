from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from torch_molecule.generator.demodiff.data import (
    ContextDataset,
    DemoDiffTokenizer,
    context_collate_fn,
)
from torch_molecule.generator.demodiff.model import DemoDiffModel


@pytest.fixture
def core_model():
    return DemoDiffModel(
        num_layer=1,
        hidden_size=32,
        num_head=4,
        mlp_ratio=2,
        diffusion_steps=2,
        drop_context=0.0,
        noise_weight=None,
        lambda_train=[1.0, 2.0, 0.5],
        dataset_info={
            "max_n_nodes": 3,
            "node_counts": [0, 1, 1, 1],
            "node_types": [0.5, 0.3, 0.2],
            "edge_types": [0.8, 0.2],
            "pos_types": [0.5, 0.3, 0.2],
            "xe_conditions": [[0.8, 0.2]] * 3,
            "ex_conditions": [[0.5, 0.3, 0.2]] * 2,
        },
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_loss_and_gradients_match_masked_cross_entropy(core_model, weighted):
    torch.manual_seed(13)
    targets = [
        F.one_hot(torch.tensor([[0, 1, 2], [1, 2, 0]]), 3).float(),
        F.one_hot(torch.randint(2, (2, 3, 3)), 2).float(),
        F.one_hot(torch.randint(3, (2, 3, 3)), 3).float(),
    ]
    targets[0][1, 2] = 0
    for truth in targets[1:]:
        truth[:, torch.arange(3), torch.arange(3)] = 0
        truth[1, 2, :] = 0
        truth[1, :, 2] = 0
    predictions = [torch.randn_like(truth, requires_grad=True) for truth in targets]
    reference_predictions = [value.detach().clone().requires_grad_() for value in predictions]
    noise = torch.tensor([[0.2], [0.8]]) if weighted else None
    pred = SimpleNamespace(**dict(zip(["X", "E", "pos"], predictions)))
    target = SimpleNamespace(**dict(zip(["X", "E", "pos"], targets)))
    actual = core_model.denoising_loss(pred, target, noise)

    expected = 0
    for scale, logits, truth in zip(core_model.lambda_train, reference_predictions, targets):
        mask = truth.sum(-1) != 0
        per_row = -(truth * logits.log_softmax(-1)).sum(-1)
        if weighted:
            sample_weights = noise.reshape((2,) + (1,) * (per_row.ndim - 1))
            per_row = per_row * sample_weights
        expected = expected + scale * per_row[mask].sum() / mask.sum()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for logits, reference in zip(predictions, reference_predictions):
        torch.testing.assert_close(logits.grad, reference.grad)


def test_single_fragment_loss_and_short_forward_are_finite(core_model):
    target = SimpleNamespace(
        X=F.one_hot(torch.tensor([[1]]), 3).float(),
        E=torch.zeros(1, 1, 1, 2),
        pos=torch.zeros(1, 1, 1, 3),
    )
    noisy = {
        "X_t": target.X,
        "E_t": target.E,
        "pos_t": target.pos,
        "node_mask": torch.ones(1, 1, dtype=torch.bool),
        "t": torch.tensor([[0.5]]),
    }
    prediction = core_model(noisy, unconditioned=True)
    assert prediction.X.shape == target.X.shape
    assert prediction.E.shape == target.E.shape
    loss = core_model.denoising_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in core_model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in core_model.parameters()
        if parameter.grad is not None
    )


def test_trained_model_uses_context_and_similarity_scores(make_model, smiles):
    torch.manual_seed(19)
    contexts = [[1], [2], [3], [0]]
    score = [[0.1], [0.3], [0.5], [0.7]]
    estimator = make_model()
    estimator.fit(smiles, contexts=contexts, score=score)
    model = estimator.model.eval()
    tokenizer = DemoDiffTokenizer()
    graphs = [tokenizer.encode(value) for value in smiles]
    dataset = ContextDataset(graphs, contexts, score)
    batch = context_collate_fn([dataset[0]])
    noisy, context = model.batch_data_process(
        batch["molecular_graphs"],
        batch["context_indicators"],
        batch["molecular_graphs"].batch,
        batch["score"],
    )
    with torch.no_grad():
        conditioned = model(noisy, context, unconditioned=False).X
        unconditional = model(noisy, context, unconditioned=True).X
        different_context = dict(context)
        different_context["score"] = context["score"].clone()
        different_context["score"][1] = 0.9
        different_score = model(noisy, different_context, unconditioned=False).X
    assert not torch.equal(conditioned, unconditional)
    assert not torch.equal(conditioned, different_score)
