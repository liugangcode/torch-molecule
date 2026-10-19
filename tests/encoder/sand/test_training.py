"""Pair-similarity training and input validation."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from torch_molecule import SANDMolecularEncoder

from .utils import PAIRS, SCORES, SEED, SMILES, local_model


@pytest.mark.parametrize("compressed", [False, True], ids=["dense", "compressed"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scratch_training_updates_weights_and_native_ema(small_params, compressed, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    params = small_params(compressed)
    torch.manual_seed(SEED)
    initial = local_model(params).state_dict()
    model = SANDMolecularEncoder(
        **params, batch_size=len(PAIRS), epochs=2, device=device, grad_clip_value=1.0
    )
    torch.manual_seed(SEED)
    with patch("torch.load", side_effect=AssertionError("Scratch training loaded weights")):
        assert model.fit(SMILES, SCORES, pairs=PAIRS) is model
    assert model.is_fitted_
    assert next(model.model.parameters()).device.type == device
    assert len(model.fitting_loss) == 2 and np.isfinite(model.fitting_loss).all()
    assert model.fitting_epoch == 1
    trained = model.model.state_dict()
    assert any(
        not torch.equal(initial[key], value.detach().cpu())
        for key, value in trained.items()
        if key.startswith("backbone.encoder")
    )
    gradients = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) for grad in gradients)
    if compressed:
        assert any(
            not torch.equal(initial[key], value.detach().cpu())
            for key, value in trained.items()
            if key.endswith("centroids")
        )
    encoded = model.encode(SMILES)
    expected = model.model.encode_smiles(SMILES, batch_size=len(PAIRS)).detach().cpu()
    assert encoded.shape == (len(SMILES), 64) and torch.isfinite(encoded).all()
    assert encoded.device.type == "cpu" and not encoded.requires_grad
    torch.testing.assert_close(encoded, expected)
    np.testing.assert_allclose(model.encode(SMILES, "np"), encoded.numpy(), atol=1e-6)


def test_refit_initializes_fresh_model(small_params):
    params = small_params()
    model = SANDMolecularEncoder(**params, batch_size=len(PAIRS), epochs=1, device="cpu")
    model.set_params(device="cpu")
    torch.manual_seed(SEED)
    model.fit(SMILES, SCORES, pairs=PAIRS)
    first_model = model.model
    first_state = copy.deepcopy(first_model.state_dict())
    first_history = model.fitting_loss.copy()
    torch.manual_seed(SEED)
    model.fit(SMILES, SCORES, pairs=PAIRS)
    assert model.model is not first_model
    assert model.fitting_loss == pytest.approx(first_history)
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, first_state[key])


def test_training_handles_singleton_tail_and_tied_score_batches(small_params):
    from torch_molecule.encoder.sand.model import SANDModel

    params = small_params()
    observed = []
    original_step = SANDModel.compute_loss

    def record_scores(module, batch, *args, **kwargs):
        observed.append(batch["score"].clone())
        return original_step(module, batch, *args, **kwargs)

    model = SANDMolecularEncoder(**params, batch_size=2, epochs=1, device="cpu")
    torch.manual_seed(SEED)
    with patch.object(SANDModel, "compute_loss", record_scores):
        model.fit(SMILES, np.array([0.1, 0.1, 0.1, 0.1, 0.9]), pairs=PAIRS[:5])
    assert len(observed) == 2
    assert sum(len(scores) for scores in observed) >= 5
    assert all(len(scores) >= 2 and scores.unique().numel() >= 2 for scores in observed)
    assert np.isfinite(model.fitting_loss).all()


def test_training_data_and_parameter_validation(small_params):
    params = small_params()
    invalid = [
        (SMILES, None, PAIRS),
        (SMILES, SCORES, None),
        ([], SCORES, PAIRS),
        (["bad-smiles"], SCORES, PAIRS),
        (SMILES, SCORES[:-1], PAIRS),
        (SMILES, SCORES[:, None], PAIRS),
        (SMILES, np.ones(len(PAIRS)), PAIRS),
        (SMILES, np.full(len(PAIRS), np.nan), PAIRS),
        (SMILES, np.full(len(PAIRS), np.inf), PAIRS),
        (SMILES, SCORES, PAIRS.ravel()),
        (SMILES, SCORES, PAIRS.astype(float)),
        (SMILES, SCORES, PAIRS - 10),
        (SMILES, SCORES, PAIRS + len(SMILES)),
        (SMILES, SCORES[:1], PAIRS[:1]),
    ]
    for molecules, scores, pairs in invalid:
        model = SANDMolecularEncoder(**params, epochs=1, device="cpu")
        with pytest.raises(ValueError):
            model.fit(molecules, scores, pairs=pairs)
        assert not model.is_fitted_
    for invalid_params in ({"batch_size": 1}, {"epochs": 0}, {"learning_rate": 0}):
        model = SANDMolecularEncoder(**params, epochs=1, device="cpu")
        model.set_params(**invalid_params)
        with pytest.raises(ValueError):
            model.fit(SMILES, SCORES, pairs=PAIRS)
