# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


class MaskedDiffusion:
    """GenMol's masked diffusion objective and confidence sampler."""

    def __init__(
        self, mask_index, num_classes, antithetic_sampling=True, sampling_eps=1e-3, noise_eps=1e-3
    ):
        self.mask_index = mask_index
        self.num_classes = num_classes
        self.antithetic_sampling = antithetic_sampling
        self.sampling_eps = sampling_eps
        self.noise_eps = noise_eps

    def sample_time(self, n_samples, device="cpu", generator=None):
        time = torch.rand(n_samples, device=device, generator=generator)
        if self.antithetic_sampling:
            offset = torch.arange(n_samples, device=device) / n_samples
            time = (time / n_samples + offset) % 1
            time = (1 - self.sampling_eps) * time + self.sampling_eps
        return time.clamp_min(torch.finfo(time.dtype).eps)

    def _sigma(self, time):
        return -torch.log1p(-(1 - self.noise_eps) * time)

    def forward_process(self, input_ids, time, generator=None):
        sigma = self._sigma(time.to(input_ids.device))
        p_mask = (1 - torch.exp(-sigma))[:, None]
        mask_indices = (
            torch.rand(input_ids.shape, device=input_ids.device, generator=generator) < p_mask
        )
        return torch.where(mask_indices, self.mask_index, input_ids)

    def _subs_parameterization(self, logits, xt):
        logits = logits.clone()
        logits[..., self.mask_index] += -1000000.0
        logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = xt != self.mask_index
        logprobs[unmasked_indices] = -1000000.0
        logprobs[unmasked_indices, xt[unmasked_indices]] = 0
        return logprobs

    def loss(self, logits, input_ids, xt, time, mask=None, global_mean=True):
        logprobs = self._subs_parameterization(logits, xt)
        log_p_theta = logprobs.gather(-1, input_ids[..., None]).squeeze(-1)
        time = time.to(input_ids.device)
        sigma = self._sigma(time)
        dsigma = (1 - self.noise_eps) / (1 - (1 - self.noise_eps) * time)
        loss = -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
        if global_mean:
            if mask is not None:
                return (loss * mask).sum() / mask.sum()
            return loss.sum() / logits.size(1)
        if mask is not None:
            return (loss * mask).sum(-1) / mask.sum(-1)
        return loss.sum(-1) / logits.size(1)

    def get_num_steps_confidence(self, xt):
        return int((xt == self.mask_index).sum(-1).max().item())

    def step_confidence(
        self, logits, xt, curr_step, num_steps, temperature=1.0, randomness=1.0, generator=None
    ):
        if temperature <= 0 or randomness < 0:
            raise ValueError("temperature must be positive and randomness must be nonnegative.")
        xt = xt.clone()
        log_p_x0 = self._subs_parameterization(logits, xt)
        probs = torch.softmax(log_p_x0 / temperature, dim=-1)
        distribution = torch.distributions.Categorical(probs=probs)
        if generator is None:
            preds = distribution.sample()
        else:
            preds = torch.multinomial(
                distribution.probs.reshape(-1, self.num_classes),
                1,
                replacement=True,
                generator=generator,
            ).reshape(xt.shape)
        confidence = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
        ratio = curr_step / (num_steps - 1)
        uniform = torch.rand(xt.shape, device=logits.device, generator=generator)
        gumbel_sample = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(uniform.dtype).tiny)))
        gumbel_noise = gumbel_sample * randomness * (1 - ratio)
        confidence = torch.log(confidence) + gumbel_noise
        mask = xt == self.mask_index
        confidence[~mask] = -torch.inf
        confidence_threshold = torch.topk(confidence, k=1, dim=-1).values
        to_replace = mask & (confidence >= confidence_threshold)
        xt[to_replace] = preds[to_replace]
        return xt
