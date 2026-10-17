"""
GNN layer modules
"""

from typing import Literal, Optional

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv, GINEConv, GPSConv, LayerNorm, PairNorm

from .mlp import MLP


class GNN(nn.Module):
    """
    Graph Neural Network (GNN) base class.
    """

    _NORM_TYPE = Optional[Literal["pair", "layer"]]

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        normalization: _NORM_TYPE = "pair",
        residual: bool = True,
    ) -> None:
        """Initialise the GNN base.

        Args:
            hidden_dim: Hidden feature dimension shared across all layers.
            num_layers: Number of message-passing layers.
            normalization (_NORM_TYPE): Normalisation after each layer —
                ``"pair"`` (PairNorm), ``"layer"`` (LayerNorm), or ``None``.
                Defaults to ``"pair"``.
            residual (bool): Whether to add residual connections. Defaults to True.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.normalization = normalization
        self.residual = residual
        self.conv_layers = self.get_conv_layers()
        self.norm_layers = self.get_norm_layers()

    def get_conv_layers(self) -> nn.ModuleList:
        """Build and return the list of convolution layers. Must be overridden."""
        raise NotImplementedError

    def get_norm_layers(self) -> Optional[nn.ModuleList]:
        """Build and return the normalisation layers, or ``None`` if not used."""
        if self.normalization is None:
            return None
        elif self.normalization == "pair":
            norm_layer = PairNorm
            nl_kwargs = {"scale": 1.0, "scale_individually": False}
        elif self.normalization == "layer":
            norm_layer = LayerNorm
            nl_kwargs = {
                "in_channels": self.hidden_dim,
                "affine": True,
                "mode": "graph",
            }
        else:
            raise ValueError("Normalization must be one of ['pair', 'layer']")

        norms = nn.ModuleList([norm_layer(**nl_kwargs) for _ in range(self.num_layers)])
        return norms

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        """Run all message-passing layers with optional normalisation and residual.

        Args:
            x (Tensor): Node feature matrix of shape ``(N, hidden_dim)``.
            edge_index (Tensor): Graph connectivity in COO format, shape ``(2, E)``.
            edge_attr (Optional[Tensor]): Edge features of shape ``(E, hidden_dim)``.
            batch (Optional[Tensor]): Per-node graph-assignment vector.
            batch_size (Optional[int]): Total number of graphs in the batch.

        Returns:
            Tensor: Updated node features of shape ``(N, hidden_dim)``.
        """
        xin = x
        for i in range(self.num_layers):
            if edge_attr is not None:
                xout = self.conv_layers[i](
                    x=xin, edge_index=edge_index, edge_attr=edge_attr
                )
            else:
                xout = self.conv_layers[i](x=xin, edge_index=edge_index)
            if self.normalization:
                xout = self.norm_layers[i](x=xout, batch=batch, batch_size=batch_size)
            if self.residual:
                xout = xout + xin
            xin = xout
        return xout


class GINELayers(GNN):
    """GINE-based GNN layers.

    Uses the Graph Isomorphism Network with Edge features (GINE) convolution from
    "Strategies for Pre-training Graph Neural Networks" (Hu et al., ICLR 2020).
    """
    def __init__(
        self,
        dim: int,
        num_layers: int,
        normalization: GNN._NORM_TYPE = "pair",
        residual: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialise GINE layers.

        Args:
            dim (int): Feature dimension for all layers.
            num_layers (int): Number of GINE convolution layers.
            normalization (GNN._NORM_TYPE): Normalisation type. Defaults to ``"pair"``.
            residual (bool): Whether to add residual connections. Defaults to True.
        """
        super().__init__(
            hidden_dim=dim,
            num_layers=num_layers,
            normalization=normalization,
            residual=residual,
        )

    def get_conv_layers(self) -> nn.ModuleList:
        """Build GINE convolution layers with inner MLP networks."""
        conv_layers = nn.ModuleList(
            [
                GINEConv(
                    nn=MLP(
                        in_dim=self.hidden_dim,
                        out_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        num_layers=3,
                        activation=nn.functional.silu,
                        norm=None,
                        raw_last_layer=False,
                    ),
                    edge_dim=None,
                )
                for _ in range(self.num_layers)
            ]
        )
        return conv_layers


class GATv2Layers(GNN):
    """GATv2-based GNN layers.

    Uses the dynamic attention mechanism from
    "How Attentive are Graph Attention Networks?" (Brody et al., ICLR 2022).
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int = 4,
        normalization: GNN._NORM_TYPE = "pair",
        residual: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialise GATv2 layers.

        Args:
            dim (int): Feature dimension for all layers.
            num_layers (int): Number of GATv2 convolution layers.
            num_heads (int): Number of attention heads. Defaults to 4.
            normalization (GNN._NORM_TYPE): Normalisation type. Defaults to ``"pair"``.
            residual (bool): Whether to add residual connections. Defaults to True.
        """
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        super().__init__(
            hidden_dim=dim,
            num_layers=num_layers,
            normalization=normalization,
            residual=residual,
        )

    def get_conv_layers(self) -> nn.ModuleList:
        """Build GATv2 convolution layers with multi-head attention."""
        head_dim = self.hidden_dim // self.num_heads
        conv_layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=self.hidden_dim,
                    out_channels=head_dim,
                    heads=self.num_heads,
                    concat=True,
                    edge_dim=self.hidden_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        return conv_layers


class GPSLayers(GNN):
    """GPS (General, Powerful, Scalable) Graph Transformer layers.

    Combines a local MPNN with global multi-head self-attention, from
    "Recipe for a General, Powerful, Scalable Graph Transformer"
    (Rampasek et al., NeurIPS 2022).
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int = 4,
        normalization: GNN._NORM_TYPE = "pair",
        residual: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialise GPS Graph Transformer layers.

        Args:
            dim (int): Feature dimension for all layers.
            num_layers (int): Number of GPS layers.
            num_heads (int): Number of attention heads. Defaults to 4.
            normalization (GNN._NORM_TYPE): Normalisation type (unused; GPS has
                internal LayerNorm). Defaults to ``"pair"``.
            residual (bool): Residual flag (unused; GPS handles residuals
                internally). Defaults to True.
        """
        self.num_heads = num_heads
        super().__init__(
            hidden_dim=dim,
            num_layers=num_layers,
            normalization=normalization,
            residual=residual,
        )

    def get_conv_layers(self) -> nn.ModuleList:
        """Build GPS convolution layers combining GINE and multi-head self-attention."""
        conv_layers = nn.ModuleList(
            [
                GPSConv(
                    channels=self.hidden_dim,
                    conv=GINEConv(
                        nn=MLP(
                            in_dim=self.hidden_dim,
                            out_dim=self.hidden_dim,
                            hidden_dim=self.hidden_dim,
                            num_layers=3,
                            activation=nn.functional.silu,
                            norm=None,
                            raw_last_layer=False,
                        ),
                        edge_dim=None,
                    ),
                    heads=self.num_heads,
                )
                for _ in range(self.num_layers)
            ]
        )
        return conv_layers

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        """Run GPS layers; residual and norm are handled internally by GPSConv.

        Args:
            x (Tensor): Node feature matrix of shape ``(N, hidden_dim)``.
            edge_index (Tensor): Graph connectivity in COO format, shape ``(2, E)``.
            edge_attr (Optional[Tensor]): Edge features of shape ``(E, hidden_dim)``.
            batch (Optional[Tensor]): Per-node graph-assignment vector.
            batch_size (Optional[int]): Unused; kept for interface compatibility.

        Returns:
            Tensor: Updated node features of shape ``(N, hidden_dim)``.
        """
        # GPSConv handles its own residual + norm internally,
        # so we skip the base class residual/norm logic.
        for i in range(self.num_layers):
            x = self.conv_layers[i](
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
            )
        return x

    def get_norm_layers(self) -> Optional[nn.ModuleList]:
        """Return ``None``; GPSConv manages its own internal LayerNorm."""
        # GPSConv includes internal LayerNorm, so no external norm needed.
        return None
