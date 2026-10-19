"""Shared SAND fixtures."""

import pytest
import torch

from torch_molecule import SANDMolecularEncoder

from .utils import PAIRS, SCORES, SMILES


@pytest.fixture
def small_params():
    def make(compressed=False):
        return {
            "num_layer": 2,
            "hidden_size": 64,
            "emb_dim": 64,
            "quantize_emb": compressed,
            "num_pq_quantizers": 4,
            "ivf_codebook_size": 64,
            "warmup_iters": 0,
        }

    return make


@pytest.fixture
def dense_params():
    return dict(
        num_layer=2,
        hidden_size=32,
        emb_dim=32,
        quantize_emb=False,
        num_pq_quantizers=4,
        ivf_codebook_size=16,
        batch_size=8,
        epochs=1,
        warmup_iters=0,
        device="cpu",
    )


@pytest.fixture
def fitted(dense_params):
    torch.manual_seed(7)
    return SANDMolecularEncoder(**dense_params).fit(SMILES, SCORES, pairs=PAIRS)
