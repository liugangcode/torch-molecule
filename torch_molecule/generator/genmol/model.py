# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. Licensed under Apache-2.0.
"""GenMol's bidirectional masked-language backbone."""

from contextlib import contextmanager
from typing import Dict

import torch
from torch import nn

from .diffusion import MaskedDiffusion
from .ema import ExponentialMovingAverage


class GenMolModel(nn.Module):
    def __init__(
        self,
        num_layer,
        num_head,
        hidden_size,
        intermediate_size,
        max_len,
        dropout,
        attention_dropout,
        hidden_act,
        layer_norm_eps,
        initializer_range,
        type_vocab_size,
        ema_decay,
        vocab_size,
        pad_token_id,
    ):
        super().__init__()
        try:
            from transformers import BertConfig, BertForMaskedLM
        except ImportError as exc:
            raise ImportError('GenMol requires: pip install "torch-molecule[genmol]"') from exc

        self.backbone = BertForMaskedLM(
            BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_layer,
                num_attention_heads=num_head,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_len,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=attention_dropout,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
                initializer_range=initializer_range,
                type_vocab_size=type_vocab_size,
                pad_token_id=pad_token_id,
                position_embedding_type="absolute",
            )
        )
        self.ema = (
            ExponentialMovingAverage(self.backbone.parameters(), decay=ema_decay)
            if ema_decay > 0
            else None
        )

    def forward(self, input_ids, attention_mask=None):
        return self.backbone(input_ids, attention_mask=attention_mask).logits

    def compute_loss(
        self,
        batched_data: Dict[str, torch.Tensor],
        diffusion: MaskedDiffusion,
        global_mean: bool = True,
    ) -> torch.Tensor:
        clean, mask = batched_data["input_ids"], batched_data["attention_mask"]
        time = diffusion.sample_time(clean.shape[0], device=clean.device)
        noisy = diffusion.forward_process(clean, time)
        return diffusion.loss(
            self(noisy, mask), clean, noisy, time, mask=mask, global_mean=global_mean
        ).mean()

    def get_extra_state(self):
        return {"ema": self.ema.state_dict() if self.ema is not None else None}

    def set_extra_state(self, state):
        saved = state["ema"]
        if (saved is None) != (self.ema is None):
            raise ValueError("Checkpoint EMA state does not match ema_decay.")
        if saved is not None:
            parameters = list(self.backbone.parameters())
            shadows = saved["shadow_params"]
            if len(shadows) != len(parameters) or any(
                shadow.shape != parameter.shape for shadow, parameter in zip(shadows, parameters)
            ):
                raise ValueError("Checkpoint EMA shapes do not match the backbone.")
            self.ema.load_state_dict(saved)
            self.ema.move_shadow_params_to_device(next(self.parameters()).device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.ema is not None:
            self.ema.shadow_params = [fn(tensor) for tensor in self.ema.shadow_params]
        return self

    @contextmanager
    def averaged_parameters(self):
        if self.ema is None:
            yield
            return
        self.ema.store(self.backbone.parameters())
        self.ema.copy_to(self.backbone.parameters())
        try:
            yield
        finally:
            self.ema.restore(self.backbone.parameters())
            self.ema.collected_params = []
