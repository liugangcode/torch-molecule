"""
Graph aggregation modules
"""

from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax


class SoftAttentionAggregation(Aggregation):
    """
    Soft attention Pooling as proposed in Li et al. (2015)
    "Gated Graph Sequence Neural Networks" <https://arxiv.org/abs/1511.05493>
    """

    def __init__(self, dim: int, node_activation: Callable = nn.Tanh()):
        super().__init__()
        self.node_activation = node_activation

        self.node_net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1), nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of this module."""
        reset(self.node_net)
        reset(self.gate_net)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        """Pool node features into a graph-level embedding using gated soft attention.

        Args:
            x (Tensor): Node feature matrix of shape ``(N, dim)``.
            index (Optional[Tensor]): Graph assignment vector of shape ``(N,)``.
            ptr (Optional[Tensor]): Optional CSR-style pointer tensor.
            dim_size (Optional[int]): Number of graphs in the batch.
            dim (int): Dimension along which to aggregate. Defaults to ``-2``.

        Returns:
            Tensor: Graph-level embeddings of shape ``(num_graphs, dim)``.
        """
        if index is None:
            index = torch.zeros(size=(x.size(0),), device=x.device, dtype=torch.long)

        if dim_size is None:
            dim_size = int(index.max()) + 1

        gate = self.gate_net(x)
        x = self.node_activation(self.node_net(x))
        x = gate * x
        x = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")
        return x


class SoftMaxAttentionAggregation(Aggregation):
    """
    Softmax Attention Pooling as proposed in Li et al. (2019)
    "Graph Matching Networks for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>
    """

    def __init__(self, dim: int):
        super().__init__()

        self.node_net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

        self.gate_net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of this module."""
        reset(self.node_net)
        reset(self.gate_net)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        """Pool node features into a graph-level embedding using softmax attention.

        Args:
            x (Tensor): Node feature matrix of shape ``(N, dim)``.
            index (Optional[Tensor]): Graph assignment vector of shape ``(N,)``.
            ptr (Optional[Tensor]): Optional CSR-style pointer tensor.
            dim_size (Optional[int]): Number of graphs in the batch.
            dim (int): Dimension along which to aggregate. Defaults to ``-2``.

        Returns:
            Tensor: Graph-level embeddings of shape ``(num_graphs, dim)``.
        """
        if index is None:
            index = torch.zeros(size=(x.size(0),), device=x.device, dtype=torch.long)

        if dim_size is None:
            dim_size = int(index.max()) + 1
        gate = self.gate_net(x)

        gate = softmax(gate, index, dim=0)
        x = self.node_net(x)
        x = gate * x
        x = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")
        return x


class GatedLinearAggregation(Aggregation):
    """
    Gated Linear Unit Pooling.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 2 * dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of this module."""
        self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        """Pool node features into a graph-level embedding using a gated linear unit.

        Args:
            x (Tensor): Node feature matrix of shape ``(N, dim)``.
            index (Optional[Tensor]): Graph assignment vector of shape ``(N,)``.
            ptr (Optional[Tensor]): Optional CSR-style pointer tensor.
            dim_size (Optional[int]): Number of graphs in the batch.
            dim (int): Dimension along which to aggregate. Defaults to ``-2``.

        Returns:
            Tensor: Graph-level embeddings of shape ``(num_graphs, dim)``.
        """
        if index is None:
            index = torch.zeros(size=(x.size(0),), device=x.device, dtype=torch.long)

        if dim_size is None:
            dim_size = int(index.max()) + 1

        gate, x = self.lin(x).chunk(2, dim=-1)
        x = gate * x
        x = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")
        return x
