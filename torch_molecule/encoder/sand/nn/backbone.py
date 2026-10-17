"""Backbone module combining the molecular encoder with IVFPQ quantization."""
from typing import Any, Dict
import torch
from .encoder import Encoder, GNN_TYPE
from .quantizer import IVFPQ


class IVFPQEncoderBackbone(torch.nn.Module):
    """Molecular encoder backbone with optional IVF+PQ vector quantization.

    Encodes a molecular graph with a GNN-based encoder and optionally passes the
    resulting embedding through an IVFPQ quantizer, storing all intermediate
    tensors needed for computing losses and for the explicit EMA codebook update.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        encoder_hidden_dim: int = 1024,
        encoder_num_layers: int = 7,
        num_atom_features: int = 62,
        num_edge_features: int = 16,
        normalize_shared_emb: bool = True,
        num_ivf_quantizers: int = 2,
        num_pq_quantizers: int = 4,
        fp_sim_factor: float = 1.0,
        shape_sim_factor: float = 1.0,
        quantize_emb: bool = True,
        ivf_codebook_size: int = 256,
        ivf_threshold_ema_dead_code: float = 0,
        ivf_ema_decay: float = 0.8,
        encoder_type: GNN_TYPE = "gine",
    ) -> None:
        """Initialise the backbone.

        Args:
            emb_dim (int): Embedding dimension. Defaults to 256.
            encoder_hidden_dim (int): Hidden dimension of the GNN encoder.
                Defaults to 1024.
            encoder_num_layers (int): Number of GNN layers. Defaults to 7.
            num_atom_features (int): Number of distinct atom-feature IDs.
                Defaults to 62.
            num_edge_features (int): Number of distinct edge-feature IDs.
                Defaults to 16.
            normalize_shared_emb (bool): Whether to L2-normalise the shared
                embedding (and the quantized reconstruction). Defaults to True.
            num_ivf_quantizers (int): Number of IVF sub-quantizers. Defaults to 2.
            num_pq_quantizers (int): Number of PQ sub-quantizers. Defaults to 4.
            fp_sim_factor (float): Unused. Defaults to 1.0.
            shape_sim_factor (float): Unused. Defaults to 1.0.
            quantize_emb (bool): Whether to apply IVFPQ quantization.
                Defaults to True.
            ivf_codebook_size (int): IVF codebook size. Defaults to 256.
            ivf_threshold_ema_dead_code (float): EMA dead-code replacement
                threshold for IVF. Defaults to 0.
            ivf_ema_decay (float): EMA decay for IVF codebook updates.
                Defaults to 0.8.
            encoder_type (GNN_TYPE): GNN architecture for the encoder.
                Defaults to ``"gine"``.
        """

        super().__init__()
        self.quantize_emb = quantize_emb
        self.emb_dim = emb_dim
        self.normalize_shared_emb = normalize_shared_emb
        self.fp_sim_factor = fp_sim_factor
        self.shape_sim_factor = shape_sim_factor
        self.ivf_codebook_size = ivf_codebook_size
        self.num_ivf_quantizers = num_ivf_quantizers
        self.num_pq_quantizers = num_pq_quantizers

        if self.quantize_emb:
            self.emb_key = "emb_quantized"
            self.ivfpq = IVFPQ(
                input_dim=emb_dim,
                num_ivf_quantizer=num_ivf_quantizers,
                num_pq_quantizer=num_pq_quantizers,
                ivf_codebook_size=ivf_codebook_size,
                ivf_threshold_ema_dead_code=ivf_threshold_ema_dead_code,
                ivf_ema_decay=ivf_ema_decay,
            )
        else:
            self.emb_key = "emb"

        self.encoder = Encoder(
            embedding_dim=emb_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            num_atom_features=num_atom_features,
            num_edge_features=num_edge_features,
            emb_key="emb",
            gnn_type=encoder_type,
        )

    def forward_batch_ivfpq(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run IVFPQ on the raw embedding in ``batch`` and populate result keys.

        Args:
            batch (Dict[str, Any]): Batch dictionary containing ``"emb"``.

        Returns:
            Dict[str, Any]: Updated batch with ``"emb_quantized"``, ``"ivf_ids"``,
                ``"pq_res_idx"``, ``"ivf_x"``, ``"pq_x_res"``,
                ``"ivf_ids_not_merged"``, and ``"x_res"`` populated.
        """

        emb = batch["emb"]
        (
            reconstructed,
            ivf_ids,
            pq_res_idx,
            ivf_x,
            pq_x_res,
            ivf_ids_not_merged,
            x_res,
        ) = self.ivfpq(emb)
        if self.normalize_shared_emb:
            reconstructed = torch.nn.functional.normalize(reconstructed, dim=-1, p=2)
        batch["emb_quantized"] = reconstructed
        batch["ivf_ids"] = ivf_ids
        batch["pq_res_idx"] = pq_res_idx
        batch["ivf_x"] = ivf_x
        batch["pq_x_res"] = pq_x_res
        batch["ivf_ids_not_merged"] = ivf_ids_not_merged
        batch["x_res"] = x_res

        return batch

    @torch.no_grad()
    def ema_update(self, batch: Dict[str, Any]) -> None:
        """Apply the codebook EMA update from tensors cached in ``batch``.

        Called during training before backpropagation. Inputs are detached
        to avoid retaining the autograd graph.
        """
        if not self.quantize_emb:
            return
        self.ivfpq.ema_update(
            batch["emb"].detach(),
            batch["ivf_ids_not_merged"],
            batch["x_res"].detach(),
            batch["pq_res_idx"],
        )

    def encode(self, batch: Dict[str, Any], batch_key: str = "batch", normalize: bool = True) -> Dict[str, Any]:
        """Encode a molecular graph batch and optionally L2-normalise the embedding.

        Args:
            batch: Batch dictionary passed to the encoder.
            batch_key (str): Per-node graph-assignment key. Defaults to ``"batch"``.
            normalize (bool): Whether to L2-normalise ``batch["emb"]``.
                Defaults to True.

        Returns:
            Updated batch with ``"emb"`` populated.
        """
        batch = self.encoder(
            batch,
            batch_key=batch_key,
        )
        if normalize:
            batch["emb"] = torch.nn.functional.normalize(batch["emb"], dim=-1, p=2)
        return batch

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Encode and optionally quantize a molecular graph batch.

        Args:
            batch (Dict[str, Any]): Batch dictionary containing graph tensors.

        Returns:
            Dict[str, Any]: Updated batch with embedding (and quantized tensors
                if ``self.quantize_emb`` is True) populated.
        """
        batch = self.encode(batch, normalize=self.normalize_shared_emb)

        if self.quantize_emb:
            batch = self.forward_batch_ivfpq(batch)

        return batch
