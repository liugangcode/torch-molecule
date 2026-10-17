"""
Multi-embedding module
"""

import torch
import torch.nn as nn

__all__ = ["MultiEmbedding"]


class MultiEmbedding(nn.Module):
    """
    Layer for shared embedding of different features, that are reduced
    along a given dimension.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 0,
        reduce: str = "sum",
        dim=1,
        *args,
        **kwargs,
    ) -> None:
        """Initialise the shared embedding layer.

        Args:
            num_embeddings (int): Vocabulary size (number of distinct feature IDs,
                excluding the padding entry added internally).
            embedding_dim (int): Dimension of each embedding vector.
            padding_idx (int): Index treated as padding (zero-embedded).
                Defaults to 0.
            reduce (str): Reduction applied across the feature dimension —
                ``"sum"`` or ``"mean"``. Defaults to ``"sum"``.
            dim (int): Dimension along which to reduce. Defaults to 1.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            *args,
            **kwargs,
        )
        self.dim = dim
        if reduce == "sum":
            self.reduce_fn = torch.sum
        elif reduce == "mean":
            self.reduce_fn = torch.mean
        else:
            raise ValueError(f"Reduce function {reduce} is not supported!")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Embed ``indices`` and reduce across the feature dimension.

        Args:
            indices (torch.Tensor): Integer index tensor of shape ``(..., F)``
                where ``F`` is the number of feature slots.

        Returns:
            torch.Tensor: Reduced embedding of shape ``(..., embedding_dim)``.
        """
        x = self.embedding(indices)
        x = self.reduce_fn(x, dim=self.dim)
        return x
