import inspect

import pytest

from torch_molecule import DemoDiffMolecularGenerator
from torch_molecule.base import BaseMolecularGenerator


def test_estimator_parameters_and_unfitted_state(make_model, tmp_path):
    model = make_model()
    assert isinstance(model, BaseMolecularGenerator)
    assert model.model is None and not model.is_fitted_
    assert model.fitting_loss == []
    parameters = inspect.signature(DemoDiffMolecularGenerator).parameters
    assert parameters.keys() <= model.get_params().keys()
    assert not {"cfg", "config", "config_path"} & parameters.keys()
    assert model.set_params(epochs=2, learning_rate=0.002) is model
    assert model.epochs == 2 and model.learning_rate == 0.002
    with pytest.raises(ValueError):
        model.set_params(unknown_parameter=True)
    with pytest.raises((ValueError, AttributeError), match="fit|train|load"):
        model.generate(1)
    with pytest.raises(ValueError, match="fitted"):
        model.save_to_local(str(tmp_path / "unfitted.pt"))
