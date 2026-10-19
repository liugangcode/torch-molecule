"""SAND trains and persists without upstream SAND or Lightning."""

import subprocess
import sys


def test_training_and_persistence_without_upstream_or_lightning(tmp_path):
    script = """
import importlib.abc
import sys
blocked = {'sand', 'lightning', 'pytorch_lightning', 'lightning_fabric',
           'faiss', 'jsonargparse', 'zarr', 'yaml'}
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Deliberately unavailable: ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockImports())
import numpy as np
import torch
from torch_molecule import SANDMolecularEncoder
from torch_molecule.encoder.sand.model import SANDModel
assert issubclass(SANDModel, torch.nn.Module)
assert all(base.__module__.split('.')[0] not in blocked for base in SANDModel.__mro__)
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCCCCCC', 'c1ccncc1',
          'C1CCCCC1', 'CC(=O)Nc1ccccc1']
pairs = [[0,1], [0,2], [0,3], [1,4], [2,5], [3,6], [4,7], [5,7]]
torch.manual_seed(7)
model = SANDMolecularEncoder(num_layer=2, hidden_size=64, emb_dim=64,
    quantize_emb=True, num_pq_quantizers=4, ivf_codebook_size=64, warmup_iters=0,
    batch_size=8, epochs=2, device='cpu')
model.fit(smiles, np.linspace(.1, .9, 8, dtype=np.float32), pairs=pairs)
assert len(model.fitting_loss) == 2 and np.isfinite(model.fitting_loss).all()
expected = model.encode(smiles)
assert expected.shape == (8, 64) and torch.isfinite(expected).all()
model.save_to_local(sys.argv[1])
restored = SANDMolecularEncoder(device='cpu')
restored.load_from_local(sys.argv[1])
torch.testing.assert_close(restored.encode(smiles), expected)
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "standalone.pt")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
