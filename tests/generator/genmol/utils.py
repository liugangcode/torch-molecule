"""Shared data and builders for GenMol tests."""

import random

import numpy as np
import torch

from torch_molecule import GenMolMolecularGenerator

SMILES = ["CCO", "CCN", "CCC", "CC(=O)O", "c1ccccc1", "c1ccncc1"]


def seed(value=7):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def small_model(**kwargs):
    params = dict(
        num_layer=1,
        num_head=2,
        hidden_size=32,
        intermediate_size=64,
        max_len=64,
        dropout=0.0,
        attention_dropout=0.0,
        ema_decay=0.9,
        batch_size=3,
        epochs=2,
        warmup_steps=0,
        learning_rate=0.002,
        device="cpu",
    )
    params.update(kwargs)
    return GenMolMolecularGenerator(**params)
