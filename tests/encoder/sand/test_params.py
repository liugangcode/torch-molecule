"""Estimator parameters, validation, and fitted-state invalidation."""

import copy

import pytest
import torch

from torch_molecule import SANDMolecularEncoder
from torch_molecule.base import BaseMolecularEncoder

from .utils import PAIRS, SCORES, SMILES


def test_estimator_parameters_and_unfitted_state():
    model = SANDMolecularEncoder(batch_size=2, epochs=2, device="cpu")
    assert isinstance(model, BaseMolecularEncoder)
    assert "checkpoint_path" not in model.get_params()
    assert "config" not in model.get_params()
    assert (model.num_layer, model.hidden_size, model.emb_dim) == (7, 1024, 512)
    assert model.quantize_emb and model.num_pq_quantizers == 32
    assert model.get_params()["epochs"] == 2
    assert model.fitting_loss == []
    assert model.set_params(batch_size=3, learning_rate=0.001) is model
    assert model.batch_size == 3 and model.learning_rate == 0.001
    with pytest.raises(ValueError):
        model.set_params(unknown_parameter=True)
    with pytest.raises(AttributeError, match="fit"):
        model.encode(SMILES)
    with pytest.raises(ValueError, match="fitted"):
        model.save_to_local("unfitted.pt")
    with pytest.raises(ValueError):
        model.fit(SMILES)


def test_explicit_parameters_reach_backbone_losses_and_schedule(small_params):
    params = small_params(compressed=True)
    params.update(
        num_layer=3,
        hidden_size=48,
        emb_dim=32,
        num_atom_features=64,
        num_edge_features=18,
        normalize_shared_emb=False,
        similarity_loss_weight=0.8,
        regularization_strength=0.7,
        uniformity_loss_weight=0.4,
        uniformity_margin=0.3,
        ivf_loss_weight=0.2,
        pq_loss_weight=0.1,
        learning_rate=0.002,
        min_lr=0.00002,
        warmup_iters=3,
        decay_iters=10,
    )
    estimator = SANDMolecularEncoder(**small_params(compressed=True), device="cpu")
    assert estimator.set_params(**params) is estimator
    assert all(estimator.get_params()[name] == value for name, value in params.items())
    model = estimator.model_class(**estimator._get_model_params())
    encoder = model.backbone.encoder.model
    assert encoder.hidden_dim == 48 and encoder.out_dim == 32
    assert encoder.gnn.num_layers == 3
    assert encoder.atom_map.embedding.num_embeddings == 65
    assert encoder.edge_map.embedding.num_embeddings == 19
    assert not model.backbone.normalize_shared_emb
    assert [output.loss_weight for output in model.outputs] == [0.8, 0.4, 0.2, 0.1]
    assert model.outputs[0].regularization_strength == 0.7
    assert model.outputs[1].margin == 0.3
    assert model.outputs[1].num_matches == estimator.uniformity_num_matches
    assert (model.lr, model.min_lr, model.warmup_iters, model.decay_iters) == (
        0.002,
        0.00002,
        3,
        10,
    )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_constructor_batch_size(batch_size):
    with pytest.raises(ValueError):
        SANDMolecularEncoder(batch_size=batch_size)


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
def test_fixed_feature_vocabularies_cannot_be_undersized(dense_params, invalid):
    with pytest.raises(ValueError, match="fixed RDKit features"):
        SANDMolecularEncoder(**dict(dense_params, **invalid))


def test_undersized_vocab_is_rejected_before_fit_initialization(dense_params, monkeypatch):
    model = SANDMolecularEncoder(**dense_params).set_params(num_atom_features=1)

    def forbidden_initialization(*args, **kwargs):
        raise AssertionError("Invalid feature vocabulary reached model allocation")

    monkeypatch.setattr(model, "_initialize_model", forbidden_initialization)
    with pytest.raises(ValueError, match="num_atom_features"):
        model.fit(SMILES, SCORES, pairs=PAIRS)
