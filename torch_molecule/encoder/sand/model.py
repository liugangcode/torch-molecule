"""SAND model and training objectives."""

import math
from typing import Optional

import torch

from .mol_utils.pytorch import MolecularGraphDataloader, smiles_to_batch
from .nn.backbone import IVFPQEncoderBackbone
from .output.embedding import (
    QuantizationOutput,
    SpearmanROutput,
    UniformityOutput,
)


class SANDModel(torch.nn.Module):
    """Molecular encoder with optional IVF/PQ quantization."""

    def __init__(
        self,
        num_layer=7,
        hidden_size=1024,
        emb_dim=512,
        encoder_type="gine",
        num_atom_features=62,
        num_edge_features=16,
        normalize_shared_emb=True,
        quantize_emb=True,
        num_ivf_quantizers=1,
        num_pq_quantizers=32,
        ivf_codebook_size=256,
        ivf_threshold_ema_dead_code=0.5,
        ivf_ema_decay=0.8,
        similarity_loss_weight=1.0,
        regularization_strength=1.0,
        uniformity_loss_weight=0.5,
        uniformity_margin=0.2,
        uniformity_num_matches=1,
        ivf_loss_weight=0.5,
        pq_loss_weight=0.25,
        learning_rate=5e-4,
        min_lr=1e-5,
        weight_decay=0.05,
        warmup_iters=10000,
        decay_iters=1_000_000,
    ):
        super().__init__()
        self.backbone = IVFPQEncoderBackbone(
            emb_dim=emb_dim,
            encoder_hidden_dim=hidden_size,
            encoder_num_layers=num_layer,
            encoder_type=encoder_type,
            num_atom_features=num_atom_features,
            num_edge_features=num_edge_features,
            normalize_shared_emb=normalize_shared_emb,
            quantize_emb=quantize_emb,
            num_ivf_quantizers=num_ivf_quantizers,
            num_pq_quantizers=num_pq_quantizers,
            ivf_codebook_size=ivf_codebook_size,
            ivf_threshold_ema_dead_code=ivf_threshold_ema_dead_code,
            ivf_ema_decay=ivf_ema_decay,
        )
        outputs = [
            SpearmanROutput(
                sim_key="score",
                emb_key=self.backbone.emb_key,
                loss_weight=similarity_loss_weight,
                regularization_strength=regularization_strength,
            ),
            UniformityOutput(
                name="uniformity" if quantize_emb else "uniformity2",
                emb_key="emb",
                loss_weight=uniformity_loss_weight,
                num_matches=uniformity_num_matches,
                margin=uniformity_margin,
            ),
        ]
        if quantize_emb:
            outputs.extend(
                [
                    QuantizationOutput(
                        loss_weight=ivf_loss_weight,
                        nq_key="emb",
                        q_key="ivf_x",
                        name="quantization_ivf",
                        use_cosine=True,
                        add_dataloader_idx_to_name=True,
                    ),
                    QuantizationOutput(
                        loss_weight=pq_loss_weight,
                        nq_key="x_res",
                        q_key="pq_x_res",
                        name="quantization_pq",
                        use_cosine=False,
                        add_dataloader_idx_to_name=True,
                    ),
                ]
            )
        self.outputs = torch.nn.ModuleList(outputs)
        self.lr = learning_rate
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters

    def forward(self, batch):
        return self.backbone(batch)

    def process_outputs(self, batch, stage="train", dataloader_idx=0):
        return sum(
            output(
                batch, stage=stage, log_func=lambda **kwargs: None, dataloader_idx=dataloader_idx
            )
            for output in self.outputs
            if output.loss_weight != 0
        )

    def compute_loss(self, batch):
        batch = self(batch)
        loss = self.process_outputs(batch, stage="train")
        # Codebook EMA uses detached tensors and runs before backward.
        if self.backbone.quantize_emb:
            self.backbone.ema_update(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.backbone.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def update_lr(self, optimizer, step):
        if step < self.warmup_iters:
            lr = self.lr * float(step + 1) / (self.warmup_iters + 1)
        elif step < self.decay_iters:
            decay_ratio = (float(step) - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.lr - self.min_lr)
        else:
            lr = self.min_lr
        for group in optimizer.param_groups:
            group["lr"] = lr

    @torch.no_grad()
    def encode_batch(self, batch, emb_key: Optional[str] = None, batch_key="batch"):
        if emb_key is None:
            emb_key = self.backbone.emb_key
        batch = batch.to(next(self.parameters()).device)
        batch = self.backbone.encode(batch, batch_key=batch_key)
        if emb_key == "emb_quantized":
            batch = self.backbone.forward_batch_ivfpq(batch)
        return batch[emb_key]

    @torch.no_grad()
    def encode_smiles(self, smiles, emb_key=None, batch_size=256, num_workers=0):
        if isinstance(smiles, str):
            smiles = [smiles]
        if len(smiles) <= batch_size:
            return self.encode_batch(smiles_to_batch(smiles), emb_key=emb_key)
        dataloader = MolecularGraphDataloader(
            smiles_list=smiles,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return torch.cat([self.encode_batch(batch, emb_key=emb_key) for batch in dataloader])
