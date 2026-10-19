"""Local and pretrained checkpoint loading."""

import copy

import pytest
import torch

from torch_molecule import SANDMolecularEncoder

from .utils import PAIRS, SCORES, SMILES, local_model


def test_trained_checkpoint_preserves_flat_attributes_and_encoding_order(small_params, tmp_path):
    params = small_params()
    params.update(
        similarity_loss_weight=0.7,
        uniformity_loss_weight=0.3,
        regularization_strength=0.8,
        min_lr=2e-5,
        warmup_iters=1,
        decay_iters=7,
    )
    model = SANDMolecularEncoder(**params, batch_size=3, epochs=2, device="cpu")
    model.fit(SMILES, SCORES, pairs=PAIRS)
    expected = model.encode(SMILES)
    saved_path = tmp_path / "sand.pt"
    model.save_to_local(str(saved_path))
    checkpoint = torch.load(saved_path, weights_only=False)
    assert "config" not in checkpoint["hyperparameters"]
    assert not any(isinstance(value, dict) for value in checkpoint["hyperparameters"].values())
    restored = SANDMolecularEncoder(device="cpu")
    restored.load_from_local(str(saved_path))
    for name, value in model.get_params().items():
        if name not in {"device", "model_name"}:
            assert restored.get_params()[name] == value, name
    restored.set_params(batch_size=1)
    torch.testing.assert_close(
        restored.encode(SMILES[::-1]), expected.flip(0), atol=1e-6, rtol=1e-5
    )


def test_pretrained_checkpoint_loading_is_explicit(small_params, tmp_path):
    params = small_params()
    native = local_model(params).eval()
    checkpoint = tmp_path / "native.ckpt"
    torch.save({"state_dict": native.state_dict()}, checkpoint)
    model = SANDMolecularEncoder(**params, batch_size=2, device="cpu")
    assert model.load_from_pretrained(str(checkpoint)) is model
    assert model.is_fitted_ and not model.model.training
    torch.testing.assert_close(model.encode(SMILES), native.encode_smiles(SMILES, batch_size=2))


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
