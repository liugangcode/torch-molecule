"""Estimator parameters, validation, and constructor layout."""

import inspect

import pytest

from torch_molecule import GenMolMolecularGenerator, MolGPTMolecularGenerator
from torch_molecule.base import BaseMolecularGenerator

from .utils import SMILES, seed, small_model


def test_parameters_and_unfitted_state(tmp_path):
    model = small_model()
    assert isinstance(model, BaseMolecularGenerator)
    assert model.model is None and not model.is_fitted_
    assert model.fitting_loss == []
    params = model.get_params()
    for name in ("num_layer", "num_head", "hidden_size", "ema_decay", "warmup_steps"):
        assert params[name] == getattr(model, name)
    assert "config" not in params and "checkpoint_path" not in params
    assert model.set_params(batch_size=2, learning_rate=0.001) is model
    assert model.batch_size == 2 and model.learning_rate == 0.001
    with pytest.raises(ValueError):
        model.set_params(unknown_parameter=True)
    with pytest.raises((ValueError, AttributeError), match="fit|train|load"):
        model.generate()
    with pytest.raises(ValueError, match="fitted"):
        model.save_to_local(str(tmp_path / "unfitted.pt"))


@pytest.mark.parametrize(
    "params",
    [
        {"batch_size": 0},
        {"num_head": 3},
        {"epochs": 0},
        {"ema_decay": 1.1},
        {"sampling_eps": -0.1},
        {"dropout": -0.1},
    ],
)
def test_invalid_configuration(params):
    with pytest.raises(ValueError):
        small_model(**params)


def test_constructor_attributes_configure_the_backbone(genmol_dependencies):
    import inspect

    model = small_model(
        intermediate_size=96, attention_dropout=0.2, hidden_act="relu", layer_norm_eps=1e-6
    )
    constructor_parameters = set(inspect.signature(GenMolMolecularGenerator).parameters)
    assert constructor_parameters <= model.get_params().keys()
    assert not constructor_parameters & {"config", "cfg", "config_path"}
    model.fit(SMILES)
    backbone = model.model.backbone.config
    assert backbone.num_hidden_layers == model.num_layer
    assert backbone.num_attention_heads == model.num_head
    assert backbone.hidden_size == model.hidden_size
    assert backbone.intermediate_size == model.intermediate_size
    assert backbone.attention_probs_dropout_prob == model.attention_dropout
    assert backbone.hidden_act == model.hidden_act
    assert backbone.layer_norm_eps == model.layer_norm_eps


def test_parameter_changes_validate_and_invalidate_fitted_architecture(genmol_dependencies):
    seed()
    model = small_model()
    model.fit(SMILES)
    with pytest.raises(ValueError):
        model.set_params(hidden_size=33)
    assert model.hidden_size == 32 and model.is_fitted_
    model.set_params(learning_rate=0.001)
    assert model.is_fitted_
    model.set_params(hidden_size=64)
    assert not model.is_fitted_
    with pytest.raises(ValueError, match="fitted"):
        model.generate()


def test_constructor_accepts_the_shared_generator_argument_layout():
    for generator in (MolGPTMolecularGenerator, GenMolMolecularGenerator):
        parameters = inspect.signature(generator).parameters
        keyword_only = {
            name
            for name, parameter in parameters.items()
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY
        }
        assert keyword_only == {"device", "model_name"}
        model = generator(1, 2, 32, device="cpu")
        assert (model.num_layer, model.num_head, model.hidden_size) == (1, 2, 32)
        assert model.get_params()["hidden_size"] == 32
