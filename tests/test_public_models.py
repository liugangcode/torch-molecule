"""Small CPU regressions for established public model workflows."""

import random

import numpy as np
import pytest
import torch
from rdkit import Chem

from torch_molecule import (
    AttrMaskMolecularEncoder,
    GNNMolecularPredictor,
    GraphGAMolecularGenerator,
    GraphMAEMolecularEncoder,
)


SMILES = ["CCO", "CCN", "CCC", "CCCl", "CC=O", "c1ccccc1"]


@pytest.fixture(autouse=True)
def small_cpu_run():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    yield
    torch.set_num_threads(previous)


def test_gnn_fit_predict_and_local_checkpoint(tmp_path):
    model = GNNMolecularPredictor(
        num_layer=2,
        hidden_size=16,
        gnn_type="gin",
        drop_ratio=0.0,
        batch_size=2,
        epochs=2,
        device="cpu",
    )
    target = np.arange(len(SMILES), dtype=np.float32).reshape(-1, 1)
    assert model.fit(SMILES[:4], target[:4], X_val=SMILES[4:], y_val=target[4:]) is model
    assert model.is_fitted_ and len(model.fitting_loss) == 2
    assert np.isfinite(model.fitting_loss).all()
    prediction = model.predict(SMILES)["prediction"]
    assert prediction.shape == target.shape
    assert np.isfinite(prediction).all()

    checkpoint = str(tmp_path / "gnn.pt")
    model.save_to_local(checkpoint)
    restored = GNNMolecularPredictor(device="cpu")
    restored.load_from_local(checkpoint)
    np.testing.assert_allclose(
        restored.predict(SMILES)["prediction"], prediction, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("encoder_class", [AttrMaskMolecularEncoder, GraphMAEMolecularEncoder])
def test_encoder_fit_encode_and_local_checkpoint(encoder_class, tmp_path):
    model = encoder_class(
        num_layer=2,
        hidden_size=16,
        encoder_type="gin",
        drop_ratio=0.0,
        batch_size=3,
        epochs=2,
        device="cpu",
    )
    assert model.fit(SMILES) is model
    assert model.is_fitted_ and len(model.fitting_loss) == 2
    assert np.isfinite(model.fitting_loss).all()
    representation = model.encode(SMILES)
    assert representation.shape == (len(SMILES), 16)
    assert torch.isfinite(representation).all()
    assert not torch.allclose(representation[0], representation[-1])

    checkpoint = str(tmp_path / "encoder.pt")
    model.save_to_local(checkpoint)
    restored = encoder_class(device="cpu")
    restored.load_from_local(checkpoint)
    torch.testing.assert_close(restored.encode(SMILES), representation, rtol=1e-6, atol=1e-6)


def test_graphga_fit_and_generate_valid_molecules():
    model = GraphGAMolecularGenerator(
        population_size=4,
        offspring_size=4,
        iteration=2,
        n_jobs=1,
        device="cpu",
    )
    assert model.fit(SMILES) is model
    assert model.is_fitted_
    molecules = model.generate(num_samples=3)
    assert len(molecules) == 3
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in molecules)
