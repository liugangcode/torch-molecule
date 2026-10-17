"""Regression checks for SAND state consistency and numerical failures."""

import copy

import numpy as np
import pytest
import torch

from torch_molecule import SANDMolecularEncoder

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "CCCCCCCC",
    "c1ccncc1",
    "C1CCCCC1",
    "CC(=O)Nc1ccccc1",
]
PAIRS = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 7]])
SCORES = np.linspace(0.1, 0.9, len(PAIRS), dtype=np.float32)


@pytest.fixture
def small_params():
    return dict(
        num_layer=2,
        hidden_size=32,
        emb_dim=32,
        quantize_emb=False,
        num_pq_quantizers=4,
        ivf_codebook_size=16,
        batch_size=8,
        epochs=1,
        warmup_iters=0,
        device="cpu",
    )


@pytest.fixture
def fitted(small_params):
    torch.manual_seed(7)
    return SANDMolecularEncoder(**small_params).fit(SMILES, SCORES, pairs=PAIRS)


def test_failed_checkpoint_load_restores_fitted_encoder(fitted, tmp_path):
    previous_model = fitted.model
    previous_params = copy.deepcopy(fitted.get_params())
    previous_embeddings = fitted.encode(SMILES)
    checkpoint = {
        "model_state_dict": copy.deepcopy(fitted.model.state_dict()),
        "hyperparameters": dict(previous_params, hidden_size=16, fitting_loss=[-1.0]),
        "model_name": "corrupt",
    }
    path = tmp_path / "incompatible.pt"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="state_dict"):
        fitted.load_from_local(str(path))
    assert fitted.model is previous_model and fitted.is_fitted_
    assert fitted.get_params() == previous_params
    torch.testing.assert_close(fitted.encode(SMILES), previous_embeddings)


@pytest.mark.parametrize("changed", [{"hidden_size": 16}, {"quantize_emb": True}])
def test_structural_parameter_changes_invalidate_fitted_state(fitted, changed, tmp_path):
    assert fitted.set_params(**changed) is fitted
    assert not fitted.is_fitted_
    with pytest.raises(AttributeError, match="fit"):
        fitted.encode(SMILES)
    with pytest.raises(ValueError, match="fitted"):
        fitted.save_to_local(str(tmp_path / "inconsistent.pt"))


def test_inference_settings_and_automatic_device_preserve_fit(fitted, monkeypatch):
    expected = fitted.encode(SMILES)
    previous_model = fitted.model
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fitted.set_params(batch_size=1, emb_key="emb", device=None, hidden_size=fitted.hidden_size)
    assert fitted.device == torch.device("cpu")
    assert fitted.model is previous_model and fitted.is_fitted_
    torch.testing.assert_close(fitted.encode(SMILES), expected, atol=1e-6, rtol=1e-5)


def test_invalid_parameter_name_does_not_partially_mutate_fit(fitted):
    previous = copy.deepcopy(fitted.get_params())
    with pytest.raises(ValueError, match="unknown_parameter"):
        fitted.set_params(hidden_size=16, unknown_parameter=True)
    assert fitted.get_params() == previous and fitted.is_fitted_


@pytest.mark.parametrize("invalid", [{"num_atom_features": 61}, {"num_edge_features": 15}])
def test_fixed_feature_vocabularies_cannot_be_undersized(small_params, invalid):
    with pytest.raises(ValueError, match="fixed RDKit features"):
        SANDMolecularEncoder(**dict(small_params, **invalid))


def test_undersized_vocab_is_rejected_before_fit_initialization(small_params, monkeypatch):
    model = SANDMolecularEncoder(**small_params).set_params(num_atom_features=1)

    def forbidden_initialization(*args, **kwargs):
        raise AssertionError("Invalid feature vocabulary reached model allocation")

    monkeypatch.setattr(model, "_initialize_model", forbidden_initialization)
    with pytest.raises(ValueError, match="num_atom_features"):
        model.fit(SMILES, SCORES, pairs=PAIRS)


def test_nonfinite_gradients_abort_before_optimizer_step(small_params, monkeypatch):
    torch.manual_seed(7)
    model = SANDMolecularEncoder(
        **small_params, similarity_loss_weight=1e38, uniformity_loss_weight=0
    )

    def forbidden_step(*args, **kwargs):
        raise AssertionError("Optimizer stepped with non-finite gradients")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    with pytest.raises(FloatingPointError, match="gradients"):
        model.fit(SMILES, SCORES, pairs=PAIRS)
    assert not model.is_fitted_
    assert all(torch.isfinite(parameter).all() for parameter in model.model.parameters())
    with pytest.raises(AttributeError, match="fit"):
        model.encode(SMILES)


def test_nonfinite_optimizer_update_cannot_produce_fitted_encoder(small_params):
    torch.manual_seed(7)
    model = SANDMolecularEncoder(**small_params, learning_rate=1e20, weight_decay=2e18)
    with pytest.raises(FloatingPointError, match="after optimizer step"):
        model.fit(SMILES, SCORES, pairs=PAIRS)
    assert not model.is_fitted_
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.model.parameters()
        if parameter.grad is not None
    )
    with pytest.raises(AttributeError, match="fit"):
        model.encode(SMILES)


def test_nonfinite_adam_moments_cannot_produce_fitted_encoder(small_params):
    torch.manual_seed(7)
    model = SANDMolecularEncoder(
        **small_params, similarity_loss_weight=1e23, uniformity_loss_weight=0
    )
    with pytest.raises(FloatingPointError, match="optimizer state"):
        model.fit(SMILES, SCORES, pairs=PAIRS)
    assert not model.is_fitted_
    assert all(torch.isfinite(parameter).all() for parameter in model.model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.model.parameters()
        if parameter.grad is not None
    )


def test_overflowed_gradient_norm_aborts_before_clipping_or_step(small_params, monkeypatch):
    torch.manual_seed(7)
    model = SANDMolecularEncoder(
        **small_params, similarity_loss_weight=1e23, uniformity_loss_weight=0, grad_clip_value=1.0
    )

    def forbidden_step(*args, **kwargs):
        raise AssertionError("Optimizer stepped after an overflowed clipping norm")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    with pytest.raises(RuntimeError, match="non-finite"):
        model.fit(SMILES, SCORES, pairs=PAIRS)
    assert not model.is_fitted_
    assert all(torch.isfinite(parameter).all() for parameter in model.model.parameters())
