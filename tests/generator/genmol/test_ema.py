"""Exponential moving average of backbone weights."""

import pytest
import torch

from torch_molecule.generator.genmol.ema import ExponentialMovingAverage

from .utils import SMILES, seed, small_model


def test_ema_uses_native_update_decay_and_restores_training_weights():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 4.0]))
    ema = ExponentialMovingAverage([parameter], decay=0.9999)
    parameter.data.copy_(torch.tensor([4.0, 8.0]))
    ema.update([parameter])
    torch.testing.assert_close(
        ema.shadow_params[0], torch.tensor([2.0, 4.0]) * (2 / 11) + parameter * (9 / 11)
    )
    trained = parameter.detach().clone()
    ema.store([parameter])
    ema.copy_to([parameter])
    torch.testing.assert_close(parameter, ema.shadow_params[0])
    ema.restore([parameter])
    torch.testing.assert_close(parameter, trained)
    restored = ExponentialMovingAverage([parameter], decay=0.5)
    restored.load_state_dict(ema.state_dict())
    assert restored.num_updates == 1
    assert restored.decay == 0.9999
    torch.testing.assert_close(restored.shadow_params[0], ema.shadow_params[0])


def test_ema_context_restores_parameters_after_exception(genmol_dependencies):
    seed()
    model = small_model()
    model.fit(SMILES)
    original = [parameter.detach().clone() for parameter in model.model.backbone.parameters()]
    with pytest.raises(RuntimeError, match="interrupted"):
        with model.model.averaged_parameters():
            for parameter, shadow in zip(
                model.model.backbone.parameters(), model.model.ema.shadow_params
            ):
                torch.testing.assert_close(parameter, shadow, rtol=0, atol=0)
            raise RuntimeError("sampling interrupted")
    for parameter, expected in zip(model.model.backbone.parameters(), original):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
