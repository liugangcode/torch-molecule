"""Optional dependencies stay optional and training frameworks stay unused."""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest


def test_public_import_does_not_require_optional_packages():
    script = """
import importlib.abc
import sys
blocked = {'transformers', 'safe', 'lightning', 'pytorch_lightning', 'hydra', 'bionemo'}
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Deliberately unavailable: ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockImports())
from torch_molecule import GenMolMolecularGenerator
model = GenMolMolecularGenerator(device='cpu')
assert model.model is None
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_training_and_sampling_do_not_import_training_frameworks():
    if any(importlib.util.find_spec(name) is None for name in ("safe", "transformers")):
        pytest.skip('Install the GenMol extra with pip install ".[genmol]"')

    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.machinery
        import math
        import random
        import sys

        blocked = {
            'datasets', 'omegaconf', 'lightning', 'pytorch_lightning',
            'hydra', 'bionemo', 'evaluate', 'wandb',
        }
        attempted = []

        class BlockImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    return importlib.machinery.ModuleSpec(fullname, self)

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                attempted.append(module.__name__)
                raise AssertionError('Unexpected optional dependency: ' + module.__name__)

        sys.meta_path.insert(0, BlockImports())

        import numpy as np
        import torch
        from rdkit import Chem
        from torch_molecule import GenMolMolecularGenerator

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        torch.set_num_threads(2)
        model = GenMolMolecularGenerator(
            num_layer=1, num_head=2, hidden_size=32, intermediate_size=64,
            max_len=32, dropout=0.0, attention_dropout=0.0, ema_decay=0.9,
            batch_size=3, epochs=1, warmup_steps=0, learning_rate=0.002,
            device='cpu',
        )
        model.fit(['CCO', 'CCN', 'CCC'])
        assert model.is_fitted_
        assert len(model.fitting_loss) == 1
        assert math.isfinite(model.fitting_loss[0])

        sampled_forwards = []
        hook = model.model.backbone.register_forward_hook(
            lambda *args: sampled_forwards.append(True)
        )
        try:
            molecules = model.generate(1, min_new_tokens=1, max_attempts=1)
            assert len(molecules) <= 1
            assert all(Chem.MolFromSmiles(value) is not None for value in molecules)
        finally:
            hook.remove()
        assert sampled_forwards
        assert not attempted, attempted
        assert not any(name.split('.')[0] in blocked for name in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
