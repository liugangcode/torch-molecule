"""Masked diffusion objective and confidence sampler."""

import pytest
import torch
import torch.nn.functional as F

from torch_molecule.generator.genmol.diffusion import MaskedDiffusion


@pytest.mark.parametrize("global_mean", [True, False])
@pytest.mark.parametrize("masked_padding", [True, False])
def test_loss_and_gradient_match_weighted_masked_cross_entropy(global_mean, masked_padding):
    torch.manual_seed(47)
    diffusion = MaskedDiffusion(mask_index=4, num_classes=5)
    logits = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    original = logits.detach().clone()
    reference_logits = logits.detach().clone().requires_grad_()
    target = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    xt = torch.tensor([[4, 1, 4, 3], [3, 4, 4, 4]])
    time = torch.tensor([0.07, 0.81], dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]) if masked_padding else torch.ones_like(target)
    actual = diffusion.loss(logits, target, xt, time, mask=mask, global_mean=global_mean)
    token_loss = F.cross_entropy(
        reference_logits[..., :4].transpose(1, 2), target, reduction="none"
    )
    weighted = token_loss * xt.eq(4) * mask / time[:, None]
    expected = weighted.sum() / mask.sum() if global_mean else weighted.sum(-1) / mask.sum(-1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(logits.detach(), original, atol=0, rtol=0)
    assert torch.count_nonzero(logits.grad[..., 4]) == 0


def test_antithetic_time_covers_each_stratum():
    diffusion = MaskedDiffusion(mask_index=4, num_classes=5, sampling_eps=0.001)
    time = diffusion.sample_time(32, generator=torch.Generator().manual_seed(19))
    bins = torch.floor((time - 0.001) / (1 - 0.001) * 32).long()
    assert torch.equal(bins, torch.arange(32))


def test_forward_process_matches_loglinear_mask_probability():
    diffusion = MaskedDiffusion(mask_index=4, num_classes=5, noise_eps=0.001)
    data = torch.arange(4).repeat(2, 3)
    time = torch.tensor([0.25, 0.75])
    draws = torch.rand(data.shape, generator=torch.Generator().manual_seed(3))
    expected = torch.where(draws < ((1 - 0.001) * time)[:, None], 4, data)
    actual = diffusion.forward_process(data, time, generator=torch.Generator().manual_seed(3))
    assert torch.equal(actual, expected)
    assert torch.equal(diffusion.forward_process(data, torch.zeros(2)), data)


@pytest.mark.parametrize("local_generator", [False, True])
def test_confidence_sampling_preserves_context_and_finishes(local_generator):
    diffusion = MaskedDiffusion(mask_index=4, num_classes=5)
    xt = torch.tensor([[0, 4, 4, 4, 3], [1, 4, 3, 0, 3]])
    initial = xt.clone()
    fixed = initial != 4
    generator = torch.Generator().manual_seed(31) if local_generator else None
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0, 0.5, 3.0],
                [7.0, 0.0, -1.0, 0.5, 3.0],
                [2.0, 0.0, 4.0, 0.5, 3.0],
                [0.0, 2.0, -1.0, 0.5, 3.0],
                [2.0, 0.0, -1.0, 0.5, 3.0],
            ]
        ]
    ).repeat(2, 1, 1)
    original_logits = logits.clone()
    steps = max(diffusion.get_num_steps_confidence(xt), 2)
    for step in range(steps):
        before = (xt == 4).sum(-1)
        xt = diffusion.step_confidence(logits, xt, step, steps, randomness=0.5, generator=generator)
        assert torch.equal(xt[fixed], initial[fixed])
        assert torch.equal((xt == 4).sum(-1), (before - 1).clamp_min(0))
    assert not (xt == 4).any()
    assert torch.equal(logits, original_logits)


def test_explicit_generator_controls_all_sampling_randomness():
    diffusion = MaskedDiffusion(mask_index=4, num_classes=5)
    xt = torch.full((3, 7), 4)
    logits = torch.randn(3, 7, 5, generator=torch.Generator().manual_seed(8))
    first = diffusion.step_confidence(logits, xt, 0, 7, generator=torch.Generator().manual_seed(11))
    torch.manual_seed(923)
    second = diffusion.step_confidence(
        logits, xt, 0, 7, generator=torch.Generator().manual_seed(11)
    )
    assert torch.equal(first, second)


def test_zero_uniform_draw_does_not_break_training_or_final_sampling(monkeypatch):
    diffusion = MaskedDiffusion(mask_index=4, num_classes=5, antithetic_sampling=False)
    monkeypatch.setattr(
        torch, "rand", lambda *args, **kwargs: torch.zeros(*args, device=kwargs.get("device"))
    )
    time = diffusion.sample_time(1)
    data = torch.tensor([[0, 1, 2]])
    xt = diffusion.forward_process(data, time)
    loss = diffusion.loss(
        torch.zeros(1, 3, 5, requires_grad=True), data, xt, time, mask=torch.ones_like(data)
    )
    assert torch.isfinite(loss)
    loss.backward()
    xt = torch.tensor([[0, 4, 2]])
    sampled = diffusion.step_confidence(torch.zeros(1, 3, 5), xt, 1, 2, randomness=0)
    assert sampled[0, 1] != 4
