"""Shared data and model builders for SAND tests."""

import numpy as np
import pytest

from torch_molecule import SANDMolecularEncoder

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "CCCCCCCC",
    "c1ccncc1",
    "C1CCCCC1",
    "CC(=O)Nc1ccccc1",
]
PAIRS = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 7]])
SCORES = np.linspace(0.1, 0.9, len(PAIRS), dtype=np.float32)
SEED = 7


def local_model(params):
    estimator = SANDMolecularEncoder(**params, device="cpu")
    return estimator.model_class(**estimator._get_model_params())


def native_model(params):
    sand = pytest.importorskip("sand")
    from sand.nn.backbone import IVFPQEncoderBackbone
    from sand.output.embedding import QuantizationOutput, SpearmanROutput, UniformityOutput

    settings = SANDMolecularEncoder(**params, device="cpu").get_params()
    backbone = IVFPQEncoderBackbone(
        emb_dim=settings["emb_dim"],
        encoder_hidden_dim=settings["hidden_size"],
        encoder_num_layers=settings["num_layer"],
        encoder_type=settings["encoder_type"],
        num_atom_features=settings["num_atom_features"],
        num_edge_features=settings["num_edge_features"],
        normalize_shared_emb=settings["normalize_shared_emb"],
        quantize_emb=settings["quantize_emb"],
        num_ivf_quantizers=settings["num_ivf_quantizers"],
        num_pq_quantizers=settings["num_pq_quantizers"],
        ivf_codebook_size=settings["ivf_codebook_size"],
        ivf_threshold_ema_dead_code=settings["ivf_threshold_ema_dead_code"],
        ivf_ema_decay=settings["ivf_ema_decay"],
    )
    outputs = [
        SpearmanROutput(
            sim_key="score",
            emb_key="emb_quantized" if settings["quantize_emb"] else "emb",
            loss_weight=settings["similarity_loss_weight"],
            regularization_strength=settings["regularization_strength"],
        ),
        UniformityOutput(
            name="uniformity" if settings["quantize_emb"] else "uniformity2",
            emb_key="emb",
            loss_weight=settings["uniformity_loss_weight"],
            num_matches=settings["uniformity_num_matches"],
            margin=settings["uniformity_margin"],
        ),
    ]
    if settings["quantize_emb"]:
        outputs.extend(
            [
                QuantizationOutput(
                    loss_weight=settings["ivf_loss_weight"],
                    nq_key="emb",
                    q_key="ivf_x",
                    name="quantization_ivf",
                    use_cosine=True,
                    add_dataloader_idx_to_name=True,
                ),
                QuantizationOutput(
                    loss_weight=settings["pq_loss_weight"],
                    nq_key="x_res",
                    q_key="pq_x_res",
                    name="quantization_pq",
                    use_cosine=False,
                    add_dataloader_idx_to_name=True,
                ),
            ]
        )
    return sand.SANDModel(
        backbone=backbone,
        outputs=outputs,
        lr=settings["learning_rate"],
        min_lr=settings["min_lr"],
        weight_decay=settings["weight_decay"],
        warmup_iters=settings["warmup_iters"],
        decay_iters=settings["decay_iters"],
    )
