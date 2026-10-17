from itertools import product
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
"""IVF-PQ quantization layers used by the SAND molecular encoder.

The centroid update follows the exponential-moving-average VQ-VAE update from
van den Oord et al. (2017). Gradient transport through the hard lookup follows
the rotation construction in Fifty et al. (2024), arXiv:2410.06424.
"""


def _unit_length(vectors: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Return vectors normalized along their feature axis."""
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(epsilon)


def _transport_lookup_gradient(
    inputs: torch.Tensor, selected_centroids: torch.Tensor
) -> torch.Tensor:
    """Attach the rotation-trick gradient estimator to a hard centroid lookup."""
    original_shape = inputs.shape
    source = inputs.reshape(-1, original_shape[-1])
    target = selected_centroids.reshape_as(source)
    source_scale = source.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target_scale = target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    source_direction = (source / source_scale).detach()
    target_direction = (target / target_scale).detach()
    reflection_axis = _unit_length(source_direction + target_direction).detach()
    reflected = source - 2 * (source * reflection_axis).sum(
        dim=-1, keepdim=True
    ) * reflection_axis
    transported = reflected + 2 * (source * source_direction).sum(
        dim=-1, keepdim=True
    ) * target_direction
    return (transported * (target_scale / source_scale).detach()).reshape(original_shape)


def _reduce_across_workers(statistic: torch.Tensor) -> None:
    """Sum a codebook statistic across distributed workers when applicable."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(statistic)


class _CentroidBank(nn.Module):
    """Independent centroid tables for each product-quantization subspace.

    Inputs stay batch-major throughout: ``(batch, subspaces, subspace_dim)``.
    Codebook mutation is deliberately confined to :meth:`ema_update`.
    """

    centroids: torch.Tensor
    ema_counts: torch.Tensor
    ema_sums: torch.Tensor

    def __init__(
        self,
        num_heads: int,
        codebook_size: int,
        dim: int,
        use_cosine_sim: bool = False,
        decay: float = 0.8,
        eps: float = 1e-5,
        threshold_ema_dead_code: float = 0.0,
        rotation_trick: bool = True,
    ) -> None:
        super().__init__()
        self.num_subspaces = num_heads
        self.codebook_size = codebook_size
        self.dim = dim
        self.use_cosine_sim = use_cosine_sim
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.rotation_trick = rotation_trick

        centroids = torch.empty(num_heads, codebook_size, dim)
        nn.init.kaiming_uniform_(centroids)
        self.register_buffer("centroids", centroids)
        self.register_buffer("ema_counts", torch.ones(num_heads, codebook_size))
        self.register_buffer("ema_sums", centroids.clone())

    def _as_subspaces(self, vectors: torch.Tensor) -> torch.Tensor:
        return vectors.reshape(vectors.shape[0], self.num_subspaces, self.dim)

    def _scores(self, vectors: torch.Tensor) -> torch.Tensor:
        if self.use_cosine_sim:
            return torch.einsum(
                "nsd,skd->nsk", _unit_length(vectors), self.centroids
            )
        similarity = torch.einsum("nsd,skd->nsk", vectors, self.centroids)
        vector_norms = vectors.square().sum(dim=-1, keepdim=True)
        centroid_norms = self.centroids.square().sum(dim=-1).unsqueeze(0)
        return -(vector_norms + centroid_norms - 2 * similarity)

    def _lookup(self, assignments: torch.Tensor) -> torch.Tensor:
        batch_size = assignments.shape[0]
        tables = self.centroids.unsqueeze(0).expand(batch_size, -1, -1, -1)
        indices = assignments[..., None, None].expand(-1, -1, 1, self.dim)
        return tables.gather(dim=2, index=indices).squeeze(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vectors = self._as_subspaces(x)
        indices = self._scores(vectors).argmax(dim=-1)
        codes = self._lookup(indices)
        if self.training and x.requires_grad:
            if self.rotation_trick:
                codes = _transport_lookup_gradient(vectors, codes)
            else:
                codes = vectors + (codes - vectors).detach()
        return codes.flatten(start_dim=1), indices

    @torch.no_grad()
    def ema_update(self, x: torch.Tensor, indices: torch.Tensor) -> None:
        observations = self._as_subspaces(x)
        if self.use_cosine_sim:
            observations = _unit_length(observations)
        head_indices = indices.transpose(0, 1)
        counts = observations.new_zeros(self.num_subspaces, self.codebook_size)
        counts.scatter_add_(1, head_indices, torch.ones_like(head_indices, dtype=observations.dtype))
        sums = observations.new_zeros(self.num_subspaces, self.codebook_size, self.dim)
        sums.scatter_add_(1, head_indices[..., None].expand(-1, -1, self.dim), observations.transpose(0, 1))
        _reduce_across_workers(counts)
        _reduce_across_workers(sums)
        self.ema_counts.lerp_(counts, 1.0 - self.decay)
        self.ema_sums.lerp_(sums, 1.0 - self.decay)
        total_mass = self.ema_counts.sum(dim=-1, keepdim=True)
        stabilized_mass = (self.ema_counts + self.eps) / (total_mass + self.codebook_size * self.eps) * total_mass
        updated_centroids = self.ema_sums / stabilized_mass.unsqueeze(-1).clamp_min(self.eps)
        if self.use_cosine_sim:
            updated_centroids = _unit_length(updated_centroids)
        self.centroids.copy_(updated_centroids)
        self._replace_inactive_centroids(observations)

    def _replace_inactive_centroids(self, observations: torch.Tensor) -> None:
        if self.threshold_ema_dead_code <= 0:
            return
        inactive = self.ema_counts < self.threshold_ema_dead_code
        for subspace in range(self.num_subspaces):
            positions = inactive[subspace].nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            samples = observations[:, subspace]
            sampled_rows = torch.randint(samples.shape[0], (positions.numel(),), device=samples.device)
            replacements = samples[sampled_rows]
            self.centroids[subspace, positions] = replacements
            self.ema_counts[subspace, positions] = self.threshold_ema_dead_code
            self.ema_sums[subspace, positions] = replacements * self.threshold_ema_dead_code


class ProductQuantizer(nn.Module):
    """Product quantizer with an independent EMA codebook per subspace.

    Codebook updates are applied via :meth:`ema_update`.
    """

    def __init__(
        self,
        input_dim: int,
        num_sub_quantizer: int,
        use_cosine_sim: bool = False,
        rotation_trick: bool = True,
        codebook_size: int = 256,  # 256, 1 Byte
        threshold_ema_dead_code: float = 0.0,
        ema_decay: float = 0.8,
    ) -> None:
        """Initialise the product quantizer.

        Args:
            input_dim (int): Total input dimension (must be divisible by
                ``num_sub_quantizer``).
            num_sub_quantizer (int): Number of disjoint subspaces.
            use_cosine_sim (bool): Use cosine distance instead of Euclidean.
                Defaults to False.
            rotation_trick (bool): Use rotation-trick STE for gradients.
                Defaults to True.
            codebook_size (int): Number of centroids per subspace codebook.
                Defaults to 256.
            threshold_ema_dead_code (float): EMA dead-code replacement threshold.
                Defaults to 0.0.
            ema_decay (float): EMA decay for codebook updates. Defaults to 0.8.
        """
        super().__init__()

        assert (
            input_dim % num_sub_quantizer == 0
        ), "input_dim must be divisible by num_sub_quantizer"

        self.sub_quantizer_dim = input_dim // num_sub_quantizer
        self.num_sub_quantizer = num_sub_quantizer

        # Quantize disjoint subspaces together; update codebooks via ema_update.
        self.codebook = _CentroidBank(
            num_heads=num_sub_quantizer,
            codebook_size=codebook_size,
            dim=self.sub_quantizer_dim,
            use_cosine_sim=use_cosine_sim,
            decay=ema_decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            rotation_trick=rotation_trick,
        )

    @property
    def codebooks(self) -> torch.Tensor:
        """Per-subspace codebooks, shape ``(num_sub_quantizer, codebook_size, sub_dim)``."""
        return self.codebook.centroids

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct quantized vectors from codebook indices.

        Args:
            indices (torch.Tensor): Index tensor of shape
                ``(batch, num_sub_quantizer)``.

        Returns:
            torch.Tensor: Reconstructed tensor of shape ``(batch, input_dim)``.
        """
        # indices: (batch, num_sub_quantizer). Gather the assigned centroid from
        # each subspace codebook and concatenate back into the full dim.
        codebooks = self.codebooks  # (S, C, d)
        num_sub, _, sub_dim = codebooks.shape
        idx = indices.long().transpose(0, 1).unsqueeze(-1).expand(num_sub, -1, sub_dim)
        codes = torch.gather(codebooks, 1, idx)  # (S, batch, d)
        return codes.permute(1, 0, 2).reshape(indices.shape[0], num_sub * sub_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``x`` and return the reconstruction and per-subspace indices.

        Args:
            x (torch.Tensor): Input of shape ``(batch, input_dim)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - quantized: ``(batch, input_dim)`` reconstructed codes.
                - indices: ``(batch, num_sub_quantizer)`` codebook indices.
        """
        return self.codebook(x)

    @torch.no_grad()
    def ema_update(self, x: torch.Tensor, indices: torch.Tensor) -> None:
        """Apply the EMA codebook update for this product quantizer.

        Args:
            x (torch.Tensor): Pre-quantization inputs of shape
                ``(batch, input_dim)``.
            indices (torch.Tensor): Codebook indices of shape
                ``(batch, num_sub_quantizer)`` from the forward pass.
        """
        self.codebook.ema_update(x, indices)


class IVFPQ(nn.Module):
    """Two-stage IVF+PQ quantizer with EMA codebook updates.

    Quantizes an embedding in two steps:
    1. **IVF** coarse quantization via a :class:`ProductQuantizer`.
    2. **PQ** fine quantization of the residual ``x - ivf_x``.

    Codebook updates are applied explicitly via :meth:`ema_update`.
    """

    def __init__(
        self,
        input_dim: int,
        num_ivf_quantizer: int = 2,
        num_pq_quantizer: int = 8,
        ivf_codebook_size: int = 256,
        ivf_threshold_ema_dead_code: float = 0.0,
        ivf_ema_decay: float = 0.8,
    ) -> None:
        """Initialise the IVFPQ quantizer.

        Args:
            input_dim (int): Dimension of the input embedding.
            num_ivf_quantizer (int): Number of IVF sub-quantizers. Defaults to 2.
            num_pq_quantizer (int): Number of PQ sub-quantizers. Defaults to 8.
            ivf_codebook_size (int): IVF codebook size per sub-quantizer.
                Defaults to 256.
            ivf_threshold_ema_dead_code (float): Dead-code replacement threshold
                for the IVF codebook. Defaults to 0.0.
            ivf_ema_decay (float): EMA decay for IVF codebook updates.
                Defaults to 0.8.
        """
        super().__init__()
        self.ivf_codebook_size = ivf_codebook_size
        self.ivf = ProductQuantizer(
            input_dim=input_dim,
            num_sub_quantizer=num_ivf_quantizer,
            use_cosine_sim=num_ivf_quantizer == 1,
            rotation_trick=True,
            codebook_size=ivf_codebook_size,
            threshold_ema_dead_code=ivf_threshold_ema_dead_code,
            ema_decay=ivf_ema_decay,
        )
        self.pq = ProductQuantizer(
            input_dim=input_dim,
            num_sub_quantizer=num_pq_quantizer,
            use_cosine_sim=False,
            rotation_trick=True,
        )
        self.input_dim = input_dim
        self.num_ivf_quantizer = num_ivf_quantizer
        self.num_pq_quantizer = num_pq_quantizer

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize ``x`` with IVF coarse quantization followed by PQ residual coding.

        Args:
            x (torch.Tensor): Input embedding of shape ``(batch, input_dim)``.

        Returns:
            Tuple of ``(reconstructed, ivf_ids, pq_res_idx, ivf_x, pq_x_res,
            ivf_ids_not_merged, x_res)`` where:

            - **reconstructed**: ``(batch, input_dim)`` IVF + PQ reconstruction.
            - **ivf_ids**: ``(batch, 1)`` for one IVF sub-quantizer, otherwise
              ``(batch,)`` merged IVF code indices.
            - **pq_res_idx**: ``(batch, num_pq_quantizer)`` PQ residual indices.
            - **ivf_x**: ``(batch, input_dim)`` IVF reconstruction.
            - **pq_x_res**: ``(batch, input_dim)`` PQ residual reconstruction.
            - **ivf_ids_not_merged**: raw per-sub-quantizer IVF indices.
            - **x_res**: ``(batch, input_dim)`` residual before PQ.
        """
        ivf_x, ivf_ids_ = self.ivf(x)
        if self.num_ivf_quantizer == 1:
            ivf_ids = ivf_ids_
        else:
            ivf_ids = (ivf_ids_[:, 0] * 256 + ivf_ids_[:, 1]).long()
        x_res = x - ivf_x.detach()
        pq_x_res, pq_res_idx = self.pq(x_res)
        reconstructed = ivf_x + pq_x_res
        return (
            reconstructed,
            ivf_ids,
            pq_res_idx,
            ivf_x,
            pq_x_res,
            ivf_ids_,
            x_res,
        )

    @torch.no_grad()
    def ema_update(
        self,
        emb: torch.Tensor,
        ivf_ids_not_merged: torch.Tensor,
        x_res: torch.Tensor,
        pq_res_idx: torch.Tensor,
    ) -> None:
        """EMA-update both codebooks from cached forward tensors (train step)."""
        self.ivf.ema_update(emb, ivf_ids_not_merged)
        self.pq.ema_update(x_res, pq_res_idx)

    def extract_code_books(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the IVF and PQ codebooks as numpy arrays.

        Constructs the full IVF codebook by concatenating all combinations of
        sub-quantizer centroids (``ivf_codebook_size ** num_ivf_quantizer`` rows).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - **ivf_code_book**: shape ``(ivf_codebook_size**n_ivf, input_dim)``.
                - **pq_code_book**: shape ``(num_pq_quantizer, codebook_size, sub_dim)``.
        """

        ivf_code_books = []
        # codebooks: [num_ivf_quantizer, codebook_size, sub_dim]
        for sub_codebook in self.ivf.codebooks:
            ivf_code_books.append(sub_codebook.cpu())

        num_centroids = self.ivf_codebook_size
        indices = list(product(range(num_centroids), repeat=self.num_ivf_quantizer))

        indices_tensor = torch.tensor(
            indices
        )  # Shape: [n_combinations, num_ivf_quantizer]

        stacked_codebooks = torch.stack(
            ivf_code_books
        )  # Shape: [num_ivf_quantizer, codebook_size, sub_dim]

        # indices_tensor[:, i] selects which centroid from codebook i for each combination
        ivf_code_book = torch.cat(
            [
                stacked_codebooks[i, indices_tensor[:, i]]
                for i in range(self.num_ivf_quantizer)
            ],
            dim=-1,
        )  # Shape: [n_combinations, full_dim]
        ivf_code_book = ivf_code_book.cpu().detach().numpy()

        # codebooks: [num_pq_quantizer, codebook_size, sub_dim]
        pq_code_book = self.pq.codebooks.detach().cpu().numpy()
        return ivf_code_book, pq_code_book

