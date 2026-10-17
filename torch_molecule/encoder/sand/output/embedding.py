"""Output heads for the SAND embedding model.

Provides correlation-based losses (:class:`SpearmanROutput`,
:class:`PearsonROutput`), a quantization alignment loss
(:class:`QuantizationOutput`), and a uniformity regularisation loss
(:class:`UniformityOutput`).
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from .base import Output


def soft_rank(values: torch.Tensor, regularization_strength: float = 1.0) -> torch.Tensor:
    """Differentiable approximate rank of the last dimension of ``values``.

    For each row, returns soft ranks in ``[0.5, n - 0.5]`` (ascending: the
    smallest element gets the lowest rank), computed as
    ``rank_i = sum_j sigmoid((x_i - x_j) / regularization_strength)``.
    As ``regularization_strength -> 0`` this converges to the exact rank.

    Args:
        values: Tensor of shape ``(*, n)``. Ranking is over the last dim.
        regularization_strength: Temperature controlling how sharply pairwise
            comparisons approximate a hard `<`. Smaller is closer to exact
            ranks but with sparser/harder gradients.
    """
    diff = values.unsqueeze(-1) - values.unsqueeze(-2)  # diff[..., i, j] = x_i - x_j
    return torch.sigmoid(diff / regularization_strength).sum(dim=-1)


class UniformityOutput(Output):
    """Output that penalises embeddings for collapsing onto the same hypersphere region."""

    def __init__(
        self,
        name: str = "uniformity",
        emb_key: Optional[str] = None,
        loss_weight: float = 1.0,
        num_matches: int = 64,
        margin: float = 0.5,
        **kwargs,
    ) -> None:
        """Initialise the uniformity output.

        Args:
            name (str): Logging name for the loss.
            emb_key (Optional[str]): Batch key for the embeddings to
                regularise. Defaults to ``"emb"``.
            loss_weight (float): Weight of this loss in the total objective.
            num_matches (int): Number of match embeddings per anchor in the
                batch.
            margin (float): Cosine-similarity threshold above which a penalty
                is applied.
            **kwargs: Additional keyword arguments forwarded to
                :class:`Output`.
        """
        self.emb_key = emb_key if emb_key is not None else "emb"
        self.num_matches = num_matches
        self.margin = margin

        super().__init__(
            name=name,
            loss=self.uniformity_loss,
            loss_weight=loss_weight,
            log_on_step_stages=["train"],
            log_on_epoch_stages=["val"],
            dataloader_idx=-1,
            **kwargs,
        )

    def uniformity_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Encourage uniform distribution on hypersphere.
        Pushes embeddings to spread out evenly.
        """
        emb = batch[self.emb_key]
        emb = emb.view(-1, self.num_matches + 1, emb.size(-1))
        ref_emb = emb[:, 0]
        cosine_sim = torch.nn.functional.cosine_similarity(
            ref_emb.unsqueeze(0), ref_emb.unsqueeze(1), dim=-1
        )
        # Exclude diagonal (self-similarity)
        mask = ~torch.eye(emb.size(0), dtype=torch.bool, device=emb.device)
        cosine_sim = cosine_sim[mask]

        # Penalize high similarity (embeddings too close)
        loss = torch.relu(cosine_sim - self.margin).mean()

        return loss


class _CosineSimCorrelationOutput(Output):
    """Shared base for correlation losses between cosine sim and target sim."""

    def __init__(
        self,
        sim_key: str,
        emb_key: Optional[str] = None,
        loss_weight: float = 1.0,
        name: Optional[str] = None,
        loss: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ) -> None:
        """Initialise the base correlation output.

        Args:
            sim_key (str): Batch key for the target similarity scores.
            emb_key (Optional[str]): Batch key for the embeddings. Defaults to
                ``"emb_{sim_key}"``.
            loss_weight (float): Weight of this loss in the total objective.
            name (Optional[str]): Logging name for the loss.
            loss: Callable that computes
                ``(pred, target) -> scalar loss``.
            **kwargs: Additional keyword arguments forwarded to
                :class:`Output`.
        """
        self.sim_key = sim_key
        self.emb_key = emb_key if emb_key is not None else f"emb_{sim_key}"
        super().__init__(
            name=name,
            loss=loss,
            loss_weight=loss_weight,
            log_on_step_stages=["train"],
            log_on_epoch_stages=["val"],
            dataloader_idx=-1,
            **kwargs,
        )

    def batch_to_pred_target(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract predicted cosine similarities and target scores from a batch.

        Reshapes the flat embedding dimension into ``(batch_size, 2, emb_dim)``
        and computes pairwise cosine similarity between the first and second
        embedding in each pair.

        Args:
            batch (Dict): Batch dict containing embedding and score tensors.

        Returns:
            Tuple ``(pred, target)`` each of shape ``(1, batch_size)``.
        """
        target = batch[self.sim_key]
        emb_dim = batch[self.emb_key].size(-1)

        emb = batch[self.emb_key].view(batch.batch_size, 2, emb_dim)
        ref_emb = emb[:, 0]
        match_emb = emb[:, 1]
        pred = torch.nn.functional.cosine_similarity(ref_emb, match_emb, dim=-1)
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        return pred, target


class SpearmanROutput(_CosineSimCorrelationOutput):
    """Soft-Spearman correlation loss between cosine sim and target sim."""

    def __init__(
        self,
        sim_key: str,
        emb_key: Optional[str] = None,
        loss_weight: float = 1.0,
        regularization_strength: float = 1.0,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the Spearman correlation output.

        Args:
            sim_key (str): Batch key for the target similarity scores.
            emb_key (Optional[str]): Batch key for the embeddings.
            loss_weight (float): Weight of this loss.
            regularization_strength (float): Temperature for
                :func:`soft_rank`. Smaller values give harder rankings.
            name (Optional[str]): Logging name; defaults to
                ``"{sim_key}_r"``.
            **kwargs: Additional keyword arguments forwarded to
                :class:`_CosineSimCorrelationOutput`.
        """
        self.regularization_strength = regularization_strength
        if name is None:
            name = f"{sim_key}_r"
        super().__init__(
            sim_key=sim_key,
            emb_key=emb_key,
            loss_weight=loss_weight,
            name=name,
            loss=self.spearmanr,
            **kwargs,
        )

    def spearmanr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the negative soft Spearman correlation.

        Args:
            pred: Predicted similarity values, shape ``(1, N)``.
            target: Target similarity values, shape ``(1, N)``.

        Returns:
            Scalar loss: ``1 - soft_spearman_r(pred, target)``.
        """
        pred = soft_rank(pred, regularization_strength=self.regularization_strength)
        target = soft_rank(target, regularization_strength=self.regularization_strength)
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return 1 - (pred * target).sum()


class PearsonROutput(_CosineSimCorrelationOutput):
    """Pearson correlation loss between cosine sim and target sim."""

    def __init__(
        self,
        sim_key: str,
        emb_key: Optional[str] = None,
        loss_weight: float = 1.0,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the Pearson correlation output.

        Args:
            sim_key (str): Batch key for the target similarity scores.
            emb_key (Optional[str]): Batch key for the embeddings.
            loss_weight (float): Weight of this loss.
            name (Optional[str]): Logging name; defaults to
                ``"{sim_key}_pearson_r"``.
            **kwargs: Additional keyword arguments forwarded to
                :class:`_CosineSimCorrelationOutput`.
        """
        if name is None:
            name = f"{sim_key}_pearson_r"
        super().__init__(
            sim_key=sim_key,
            emb_key=emb_key,
            loss_weight=loss_weight,
            name=name,
            loss=self.pearsonr,
            **kwargs,
        )

    def pearsonr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the negative Pearson correlation.

        Args:
            pred: Predicted similarity values, shape ``(1, N)``.
            target: Target similarity values, shape ``(1, N)``.

        Returns:
            Scalar loss: ``1 - pearson_r(pred, target)``.
        """
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return 1 - (pred * target).sum()


class QuantizationOutput(Output):
    """Output that aligns quantized embeddings to their non-quantized counterparts."""

    def __init__(
        self,
        name: str = "quantizer",
        loss_weight: float = 1.0,
        dataloader_idx: int = -1,
        nq_key: str = "emb_non_quantized",
        q_key: str = "emb",
        use_cosine: bool = False,
        **kwargs,
    ) -> None:
        """Initialise the quantization alignment output.

        Args:
            name (str): Logging name.
            loss_weight (float): Weight of this loss.
            dataloader_idx (int): Index of the dataloader this output applies
                to.
            nq_key (str): Batch key for the non-quantized embeddings.
            q_key (str): Batch key for the quantized embeddings.
            use_cosine (bool): If ``True``, use cosine distance; otherwise
                use mean squared error.
            **kwargs: Additional keyword arguments forwarded to
                :class:`Output`.
        """
        self.nq_key = nq_key
        self.q_key = q_key
        self.use_cosine = use_cosine
        super().__init__(
            name,
            loss=self._loss,
            loss_weight=loss_weight,
            log_on_step_stages=["train"],
            log_on_epoch_stages=["val"],
            dataloader_idx=dataloader_idx,
            **kwargs,
        )

    def _loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute the alignment loss between quantized and non-quantized embeddings."""
        x = batch[self.nq_key]
        x_q = batch[self.q_key]
        if self.use_cosine:
            loss = 1 - torch.nn.functional.cosine_similarity(x, x_q, dim=-1).mean()
        else:
            loss = torch.sum((x - x_q) ** 2, dim=-1).mean()
        return loss
