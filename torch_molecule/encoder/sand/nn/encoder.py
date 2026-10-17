"""GNN-based molecular encoder with configurable backbone."""
from typing import Dict, Any, Literal

import torch

from .molecule_encoder import (
    GATv2MoleculeEncoder,
    GINEMoleculeEncoder,
    GPSMoleculeEncoder,
)

GNN_TYPE = Literal["gine", "gatv2", "gps"]

_ENCODER_CLS = {
    "gine": GINEMoleculeEncoder,
    "gatv2": GATv2MoleculeEncoder,
    "gps": GPSMoleculeEncoder,
}


class Encoder(torch.nn.Module):
    """Configurable GNN encoder that maps a molecular graph to a fixed-size embedding.

    Wraps one of :class:`GINEMoleculeEncoder`, :class:`GATv2MoleculeEncoder`, or
    :class:`GPSMoleculeEncoder` and stores the resulting embedding under a
    configurable key in the input batch dictionary.
    """

    def __init__(
        self,
        emb_key: str = "emb",
        embedding_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 3,
        dropout: float = 0.0,
        num_atom_features: int = 155,
        num_edge_features: int = 30,
        gnn_type: GNN_TYPE = "gine",
        num_heads: int = 4,
    ):
        """Initialise the encoder.

        Args:
            emb_key (str): Key under which the embedding is stored in the batch
                dictionary. Defaults to ``"emb"``.
            embedding_dim (int): Output embedding dimension. Defaults to 512.
            hidden_dim (int): Hidden dimension for the GNN and MLP. Defaults to 768.
            num_layers (int): Number of GNN message-passing layers. Defaults to 3.
            dropout (float): Dropout probability. Defaults to 0.0.
            num_atom_features (int): Number of distinct atom-feature IDs.
                Defaults to 155.
            num_edge_features (int): Number of distinct edge-feature IDs.
                Defaults to 30.
            gnn_type (GNN_TYPE): GNN architecture — ``"gine"``, ``"gatv2"``, or
                ``"gps"``. Defaults to ``"gine"``.
            num_heads (int): Number of attention heads (GATv2 and GPS only).
                Defaults to 4.
        """
        super().__init__()
        self.emb_key = emb_key

        encoder_cls = _ENCODER_CLS[gnn_type]
        encoder_kwargs = dict(
            hidden_dim=hidden_dim,
            num_atom_features=num_atom_features,
            num_edge_features=num_edge_features,
            num_gnn_layers=num_layers,
            num_mlp_layers=3,
            out_dim=embedding_dim,
            mlp_normalization="layer",
        )
        if gnn_type == "gine":
            encoder_kwargs["gnn_normalization"] = "layer"
        elif gnn_type == "gatv2":
            encoder_kwargs["gnn_normalization"] = "layer"
            encoder_kwargs["num_heads"] = num_heads
        elif gnn_type == "gps":
            encoder_kwargs["num_heads"] = num_heads

        self.model = encoder_cls(**encoder_kwargs)

    def forward(
        self,
        batch: Dict[str, Any],
        batch_key: str = "atom_part_batch",
    ) -> Dict[str, Any]:
        """Encode a molecular graph batch into fixed-size embeddings.

        Args:
            batch (Dict[str, Any]): Batch dictionary containing ``"x"``,
                ``"edge_index"``, ``"edge_attr"``, and a per-node batch-assignment
                vector under ``batch_key``.
            batch_key (str): Key of the per-node graph-assignment vector.
                Defaults to ``"atom_part_batch"``.

        Returns:
            Dict[str, Any]: Updated batch with the embedding stored under
                ``self.emb_key``.
        """
        emb = self.model(
            x=batch["x"],
            edge_index=batch["edge_index"],
            edge_attr=batch["edge_attr"],
            batch=batch[batch_key],
            batch_size=batch.get("num_graphs", batch.get("batch_size")),
        )
        batch[self.emb_key] = emb
        return batch
