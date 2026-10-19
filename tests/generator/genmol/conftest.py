"""Shared GenMol fixtures."""

import importlib.util

import pytest
import torch

from torch_molecule import GenMolMolecularGenerator
from torch_molecule.generator.genmol import modeling_genmol


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def genmol_dependencies():
    if any(importlib.util.find_spec(name) is None for name in ["safe", "transformers"]):
        pytest.skip('Install the GenMol extra with pip install ".[genmol]"')
    import safe
    import transformers


@pytest.fixture
def make_model():
    pytest.importorskip("safe")
    pytest.importorskip("transformers")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)

    def create(model_class=GenMolMolecularGenerator, **kwargs):
        params = dict(
            num_layer=1,
            num_head=2,
            hidden_size=16,
            intermediate_size=32,
            max_len=32,
            dropout=0.0,
            attention_dropout=0.0,
            batch_size=2,
            epochs=2,
            warmup_steps=0,
            device="cpu",
        )
        params.update(kwargs)
        return model_class(**params)

    yield create
    torch.set_num_threads(previous)


@pytest.fixture
def progress_bars(monkeypatch):
    bars = []

    class ProgressBar:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.updates = 0
            self.closed = False
            self.logs = []
            bars.append(self)

        def update(self, amount):
            self.updates += amount

        def set_postfix(self, values):
            self.logs.append(values)

        def close(self):
            self.closed = True

    monkeypatch.setattr(modeling_genmol, "tqdm", ProgressBar)
    return bars
