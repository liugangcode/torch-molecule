"""Embedding inference and its input validation."""

import pytest

from torch_molecule import SANDMolecularEncoder

from .utils import PAIRS, SCORES, SMILES


def test_encoding_validation_after_training(small_params):
    params = small_params()
    model = SANDMolecularEncoder(**params, epochs=1, device="cpu")
    model.fit(SMILES, SCORES, pairs=PAIRS)
    for invalid in ([], ["invalid-smiles"], [None], "CCO"):
        with pytest.raises(ValueError):
            model.encode(invalid)
    with pytest.raises(ValueError):
        model.encode(SMILES, return_type="invalid")
    model.set_params(batch_size=0)
    with pytest.raises(ValueError):
        model.encode(SMILES)
