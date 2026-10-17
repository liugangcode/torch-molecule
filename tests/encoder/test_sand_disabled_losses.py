"""Zero-weight objectives must not evaluate undefined intermediate losses."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from torch_molecule import SANDMolecularEncoder
from torch_molecule.encoder.sand.output.embedding import SpearmanROutput


def test_disabled_spearman_is_skipped_during_uniformity_training():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    pairs = np.column_stack([np.arange(len(smiles))] * 2)
    model = SANDMolecularEncoder(
        num_layer=2,
        hidden_size=32,
        emb_dim=16,
        quantize_emb=False,
        similarity_loss_weight=0.0,
        uniformity_loss_weight=0.5,
        warmup_iters=0,
        batch_size=4,
        epochs=2,
        device="cpu",
    )
    torch.manual_seed(7)
    with patch.object(
        SpearmanROutput, "forward", side_effect=AssertionError("Disabled loss was evaluated")
    ) as disabled:
        model.fit(smiles, np.linspace(0.1, 0.9, len(pairs)), pairs=pairs)
    disabled.assert_not_called()
    assert model.is_fitted_
    assert len(model.fitting_loss) == 2
    assert np.isfinite(model.fitting_loss).all()
    assert all(loss > 0 for loss in model.fitting_loss)
    gradients = [
        parameter.grad for parameter in model.model.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    assert torch.isfinite(model.encode(smiles)).all()


def test_active_spearman_still_reports_constant_prediction_failure():
    model = SANDMolecularEncoder(
        num_layer=2,
        hidden_size=32,
        emb_dim=16,
        quantize_emb=False,
        warmup_iters=0,
        batch_size=2,
        epochs=1,
        device="cpu",
    )
    torch.manual_seed(7)
    with pytest.raises(FloatingPointError, match="loss is non-finite"):
        model.fit(["CCO", "CCN"], [0.1, 0.9], pairs=[[0, 1], [0, 1]])
    assert not model.is_fitted_
