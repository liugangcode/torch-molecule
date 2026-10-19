"""GenMol backbone module."""

import copy

import pytest
import torch

from torch_molecule.generator.genmol.diffusion import MaskedDiffusion
from torch_molecule.generator.genmol.model import GenMolModel


@pytest.mark.parametrize("antithetic_sampling", [False, True])
@pytest.mark.parametrize("global_mean", [False, True])
def test_compute_loss_preserves_native_loss_and_gradients(antithetic_sampling, global_mean):
    pytest.importorskip("transformers")
    torch.manual_seed(17)
    model = GenMolModel(
        num_layer=1,
        num_head=2,
        hidden_size=16,
        intermediate_size=32,
        max_len=8,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        type_vocab_size=2,
        ema_decay=0.0,
        vocab_size=12,
        pad_token_id=3,
    )
    reference = copy.deepcopy(model)
    diffusion = MaskedDiffusion(4, 12, antithetic_sampling=antithetic_sampling)
    batch = {
        "input_ids": torch.tensor([[0, 5, 6, 7, 1], [0, 8, 1, 3, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
    }

    torch.manual_seed(29)
    actual = model.compute_loss(batch, diffusion, global_mean=global_mean)
    actual.backward()

    torch.manual_seed(29)
    clean, mask = batch["input_ids"], batch["attention_mask"]
    time = diffusion.sample_time(clean.shape[0], device=clean.device)
    noisy = diffusion.forward_process(clean, time)
    expected = diffusion.loss(
        reference(noisy, mask), clean, noisy, time, mask=mask, global_mean=global_mean
    ).mean()
    expected.backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
