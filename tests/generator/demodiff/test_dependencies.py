import subprocess
import sys
import textwrap


def test_training_and_sampling_use_existing_dependencies():
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.machinery
        import math
        import sys

        blocked = {'lightning', 'pytorch_lightning', 'hydra', 'omegaconf', 'wandb', 'torchmetrics', 'datasets'}
        attempted = []

        class BlockImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    return importlib.machinery.ModuleSpec(fullname, self)

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                attempted.append(module.__name__)
                raise AssertionError('Unexpected dependency: ' + module.__name__)

        sys.meta_path.insert(0, BlockImports())
        import torch
        from torch_molecule import DemoDiffMolecularGenerator

        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = DemoDiffMolecularGenerator(
            num_layer=1, hidden_size=32, num_head=4, diffusion_steps=2,
            context_length=64, batch_size=2, epochs=1, device='cpu',
        )
        model.fit(['CCO', 'CCN'])
        assert model.is_fitted_ and math.isfinite(model.fitting_loss[0])
        model.generate(1, num_nodes=1, max_attempts=1)
        assert not attempted, attempted
        assert not any(name.split('.')[0] in blocked for name in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
