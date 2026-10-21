import math

import numpy as np
import pytest
import torch

from torch_molecule.generator.demodiff.transformer import Transformer


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_updates_weights_with_finite_gradients(make_model, smiles, device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    observed_steps = []
    original_step = torch.optim.AdamW.step

    def monitored_step(optimizer, *args, **kwargs):
        assert optimizer.defaults["amsgrad"]
        assert optimizer.defaults["eps"] == 1e-6
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        gradients = [p.grad for p in parameters if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        previous = [p.detach().clone() for p in parameters]
        result = original_step(optimizer, *args, **kwargs)
        assert all(torch.isfinite(p).all() for p in parameters)
        observed_steps.append(
            any(not torch.equal(before, after) for before, after in zip(previous, parameters))
        )
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", monitored_step)
    torch.manual_seed(17)
    model = make_model(device=device)
    contexts = [[1], [2], [3], [0]]
    score = [[0.2], [0.4], [0.6], [0.8]]
    assert model.fit(smiles, contexts=contexts, score=score) is model
    assert model.is_fitted_
    assert len(model.fitting_loss) == model.epochs
    assert np.isfinite(model.fitting_loss).all()
    assert len(observed_steps) == model.epochs * math.ceil(len(smiles) / model.batch_size)
    assert all(observed_steps)


def test_fit_reinitializes_weights_and_history(make_model, smiles):
    model = make_model()
    model.fit(smiles)
    previous = model.model
    with torch.no_grad():
        for parameter in previous.parameters():
            parameter.fill_(42)
    assert model.fit(smiles[:2]) is model
    assert model.model is not previous
    assert len(model.fitting_loss) == 1
    assert np.isfinite(model.fitting_loss).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.model.parameters())


@pytest.mark.parametrize(
    "conditioned,drop_context,no_context_train",
    [(True, 0.0, False), (True, 1.0, False), (True, 0.0, True), (False, 0.0, False)],
)
def test_training_passes_target_one_and_similarity_scores_to_transformer(
    make_model, smiles, monkeypatch, conditioned, drop_context, no_context_train
):
    observed = []
    original_forward = Transformer.forward

    def record_forward(self, *args, **kwargs):
        score, masks = args[3:5]
        target_mask = masks["target"]
        target_width = target_mask.shape[1]
        assert torch.all(score[:, :target_width, 0][target_mask] == 1)
        observed.append(
            (score.detach()[masks["all"]].clone(), masks["all"].sum(), target_mask.sum())
        )
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(Transformer, "forward", record_forward)
    model = make_model(drop_context=drop_context, no_context_train=no_context_train)
    kwargs = (
        {"contexts": [[1], [2], [3], [0]], "score": [[0.0], [1.0], [0.0], [1.0]]}
        if conditioned
        else {}
    )
    model.fit(smiles, **kwargs)
    assert observed
    if conditioned and drop_context == 0 and not no_context_train:
        assert all(all_nodes > target_nodes for _, all_nodes, target_nodes in observed)
        assert any(torch.any(score == 0) for score, _, _ in observed)
    else:
        assert all(all_nodes == target_nodes for _, all_nodes, target_nodes in observed)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contexts": [[1], [0]]},
        {"score": [[0.2], [0.4]]},
        {"contexts": [[1], [0]], "score": [[0.2]]},
        {"contexts": [[1], [0]], "score": [[], [0.4]]},
        *[
            {"contexts": [[1], [0]], "score": [[value], [0.4]]}
            for value in [-0.01, 1.01, float("nan"), float("inf")]
        ],
    ],
)
def test_fit_rejects_missing_mismatched_or_invalid_similarity_scores(make_model, smiles, kwargs):
    model = make_model()
    with pytest.raises(ValueError, match="score"):
        model.fit(smiles[:2], **kwargs)
    assert not model.is_fitted_
