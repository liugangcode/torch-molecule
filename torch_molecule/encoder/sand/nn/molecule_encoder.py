"""
Molecule encoder modules
"""

from typing import Optional

import torch
from torch.nn import Module

from .aggregation import SoftMaxAttentionAggregation
from .embedding import MultiEmbedding
from .gnn import GATv2Layers, GINELayers, GPSLayers
from .mlp import MLP


class MoleculeEncoder(Module):
    """
    Molecule encoder module that wraps atom/edge embedding, GNN layers,
    atom-level aggregation and final MLP layers.
    """

    def __init__(
        self,
        atom_map: Module,
        edge_map: Module,
        gnn: Module,
        aggregation: Module,
        mlp: Module,
        out_nonlin: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()

        self.atom_map = atom_map
        self.edge_map = edge_map
        self.gnn = gnn
        self.aggregation = aggregation
        self.mlp = mlp
        self.out_nonlin = out_nonlin

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode a molecular graph into a fixed-size embedding.

        Args:
            x (torch.Tensor): Atom feature indices of shape ``(N,)`` or
                ``(N, F)`` depending on the embedding layer.
            edge_index (torch.Tensor): Edge index tensor of shape ``(2, E)``.
            edge_attr (torch.Tensor): Edge feature indices.
            batch (torch.Tensor): Per-node graph-assignment vector.
            batch_size (Optional[int]): Number of graphs in the batch.
                Inferred from ``batch`` if not provided.

        Returns:
            torch.Tensor: Graph-level embedding of shape
                ``(num_graphs, out_dim)``.
        """
        
        bs = batch_size if batch_size is not None else int(batch.max()) + 1
        edge_attr = self.edge_map(edge_attr)
        x = self.atom_map(x)

        x = self.gnn(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=bs,
        )
        y = self.aggregation(x, batch, dim_size=bs, dim=0)
        y = self.mlp(y)
        if self.out_nonlin:
            y = self.out_nonlin(y)
        return y


class GINEMoleculeEncoder(MoleculeEncoder):
    """
    GINE-based molecule encoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_atom_features: int,
        num_edge_features: int,
        num_gnn_layers: int,
        num_mlp_layers: int,
        out_dim: int,
        gnn_normalization: GINELayers._NORM_TYPE = "layer",
        mlp_normalization: MLP._NORM_TYPE = None,
        out_nonlin: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialise a GINE molecule encoder.

        Args:
            hidden_dim (int): Hidden dimension for GNN and MLP layers.
            num_atom_features (int): Number of distinct atom-feature IDs.
            num_edge_features (int): Number of distinct edge-feature IDs.
            num_gnn_layers (int): Number of GINE message-passing layers.
            num_mlp_layers (int): Number of MLP layers in the readout.
            out_dim (int): Output embedding dimension.
            gnn_normalization (GINELayers._NORM_TYPE): GNN normalisation type.
                Defaults to ``"layer"``.
            mlp_normalization (MLP._NORM_TYPE): MLP normalisation type.
                Defaults to ``None``.
            out_nonlin (Optional[torch.nn.Module]): Optional non-linearity
                applied to the final embedding. Defaults to ``None``.
        """
        atom_map = MultiEmbedding(
            num_embeddings=num_atom_features, embedding_dim=hidden_dim
        )
        edge_map = MultiEmbedding(
            num_embeddings=num_edge_features, embedding_dim=hidden_dim
        )
        gnn = GINELayers(
            dim=hidden_dim,
            num_layers=num_gnn_layers,
            normalization=gnn_normalization,
            residual=True,
        )
        mlp = MLP(
            in_dim=hidden_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            norm=mlp_normalization,
        )
        aggregation = SoftMaxAttentionAggregation(dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        super().__init__(
            atom_map=atom_map,
            edge_map=edge_map,
            gnn=gnn,
            aggregation=aggregation,
            mlp=mlp,
            out_nonlin=out_nonlin,
        )


class GATv2MoleculeEncoder(MoleculeEncoder):
    """GATv2-based molecule encoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_atom_features: int,
        num_edge_features: int,
        num_gnn_layers: int,
        num_mlp_layers: int,
        out_dim: int,
        num_heads: int = 4,
        gnn_normalization: GATv2Layers._NORM_TYPE = "layer",
        mlp_normalization: MLP._NORM_TYPE = None,
        out_nonlin: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialise a GATv2 molecule encoder.

        Args:
            hidden_dim (int): Hidden dimension for GNN and MLP layers.
            num_atom_features (int): Number of distinct atom-feature IDs.
            num_edge_features (int): Number of distinct edge-feature IDs.
            num_gnn_layers (int): Number of GATv2 message-passing layers.
            num_mlp_layers (int): Number of MLP layers in the readout.
            out_dim (int): Output embedding dimension.
            num_heads (int): Number of attention heads. Defaults to 4.
            gnn_normalization (GATv2Layers._NORM_TYPE): GNN normalisation type.
                Defaults to ``"layer"``.
            mlp_normalization (MLP._NORM_TYPE): MLP normalisation type.
                Defaults to ``None``.
            out_nonlin (Optional[torch.nn.Module]): Optional non-linearity
                applied to the final embedding. Defaults to ``None``.
        """
        atom_map = MultiEmbedding(
            num_embeddings=num_atom_features, embedding_dim=hidden_dim
        )
        edge_map = MultiEmbedding(
            num_embeddings=num_edge_features, embedding_dim=hidden_dim
        )
        gnn = GATv2Layers(
            dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            normalization=gnn_normalization,
            residual=True,
        )
        mlp = MLP(
            in_dim=hidden_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            norm=mlp_normalization,
        )
        aggregation = SoftMaxAttentionAggregation(dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        super().__init__(
            atom_map=atom_map,
            edge_map=edge_map,
            gnn=gnn,
            aggregation=aggregation,
            mlp=mlp,
            out_nonlin=out_nonlin,
        )


class GPSMoleculeEncoder(MoleculeEncoder):
    """GPS Graph Transformer molecule encoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_atom_features: int,
        num_edge_features: int,
        num_gnn_layers: int,
        num_mlp_layers: int,
        out_dim: int,
        num_heads: int = 4,
        mlp_normalization: MLP._NORM_TYPE = None,
        out_nonlin: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialise a GPS molecule encoder.

        Args:
            hidden_dim (int): Hidden dimension for GNN and MLP layers.
            num_atom_features (int): Number of distinct atom-feature IDs.
            num_edge_features (int): Number of distinct edge-feature IDs.
            num_gnn_layers (int): Number of GPS layers.
            num_mlp_layers (int): Number of MLP layers in the readout.
            out_dim (int): Output embedding dimension.
            num_heads (int): Number of attention heads. Defaults to 4.
            mlp_normalization (MLP._NORM_TYPE): MLP normalisation type.
                Defaults to ``None``.
            out_nonlin (Optional[torch.nn.Module]): Optional non-linearity
                applied to the final embedding. Defaults to ``None``.
        """
        atom_map = MultiEmbedding(
            num_embeddings=num_atom_features, embedding_dim=hidden_dim
        )
        edge_map = MultiEmbedding(
            num_embeddings=num_edge_features, embedding_dim=hidden_dim
        )
        gnn = GPSLayers(
            dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            residual=True,
        )
        mlp = MLP(
            in_dim=hidden_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            norm=mlp_normalization,
        )
        aggregation = SoftMaxAttentionAggregation(dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        super().__init__(
            atom_map=atom_map,
            edge_map=edge_map,
            gnn=gnn,
            aggregation=aggregation,
            mlp=mlp,
            out_nonlin=out_nonlin,
        )
