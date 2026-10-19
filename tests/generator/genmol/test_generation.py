"""Sampling, scaffold completion, and generation arguments."""

import warnings

import pytest
import torch
from rdkit import Chem

from torch_molecule import GenMolMolecularGenerator

from .utils import SMILES, seed, small_model


def test_invalid_generation_arguments_are_rejected(genmol_dependencies):
    seed()
    model = small_model(epochs=1)
    model.fit(SMILES)
    assert model.generate(n_samples=0) == []
    for params in [
        {"n_samples": -1},
        {"temperature": 0},
        {"temperature": float("nan")},
        {"randomness": -1},
        {"min_new_tokens": 0},
        {"max_attempts": 0},
        {"context_mask_ratio": 1.1},
        {"scaffold": "not-a-molecule"},
        {"scaffold": "*"},
        {"scaffold": "CC.CC"},
    ]:
        with pytest.raises(ValueError):
            model.generate(**params)


def test_trained_generation_scaffolds_and_seeded_checkpoint_roundtrip(
    tmp_path, genmol_dependencies
):
    seed(41)
    model = small_model(epochs=60, batch_size=6, learning_rate=0.005, max_len=32)
    model.fit(["CCC", "CCCC", "CCO", "CCN", "CCCl", "CC(C)C"])
    seed(83)
    samples = model.generate(n_samples=6, min_new_tokens=1, max_attempts=20)
    assert len(samples) == 6
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in samples)
    constrained = model.generate(n_samples=3, scaffold="CC", min_new_tokens=2, max_attempts=20)
    query = Chem.MolFromSmiles("CC")
    assert all(Chem.MolFromSmiles(smiles).HasSubstructMatch(query) for smiles in constrained)
    path = tmp_path / "generation.pt"
    model.save_to_local(path)
    restored = GenMolMolecularGenerator(device="cpu")
    restored.load_from_local(path)
    seed(83)
    assert restored.generate(n_samples=6, min_new_tokens=1, max_attempts=20) == samples


@pytest.mark.parametrize("decoded", [[None, None, None], ["CC", None, None], ["CC", "CO", "CN"]])
def test_sampling_returns_available_molecules_and_warns_on_shortfall(
    decoded, monkeypatch, genmol_dependencies
):
    seed()
    model = small_model(epochs=1)
    model.fit(SMILES)
    model.model.train()
    original = [parameter.detach().clone() for parameter in model.model.parameters()]
    outputs = iter(decoded)
    monkeypatch.setattr(
        "torch_molecule.generator.genmol.modeling_genmol.decode_safe",
        lambda *args, **kwargs: next(outputs),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        samples = model.generate(n_samples=3, min_new_tokens=1, max_attempts=1)
    assert samples == [smiles for smiles in decoded if smiles is not None]
    if len(samples) < 3:
        assert len(caught) == 1 and caught[0].category is UserWarning
        assert f"generated {len(samples)} of 3" in str(caught[0].message)
    else:
        assert not caught
    assert model.model.training
    for parameter, expected in zip(model.model.parameters(), original):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_sampling_failure_restores_training_weights_and_mode(monkeypatch, genmol_dependencies):
    seed()
    model = small_model()
    model.fit(SMILES)
    original = [parameter.detach().clone() for parameter in model.model.parameters()]
    model.model.train()

    def fail_sampling(*args, **kwargs):
        raise RuntimeError("sampling interrupted")

    monkeypatch.setattr(model, "_sample_tokens", fail_sampling)
    with pytest.raises(RuntimeError, match="sampling interrupted"):
        model.generate(n_samples=1, min_new_tokens=1)
    assert model.model.training
    for parameter, expected in zip(model.model.parameters(), original):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
