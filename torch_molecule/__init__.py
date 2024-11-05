__version__ = "1.0.0.dev0"

"""
torch_molecule.predictor module
Provides molecular property prediction functionality
"""

# from torch_molecule.base import BaseMolecularPredictor
from torch_molecule.predictor.grea import GREAMolecularPredictor
from torch_molecule.predictor.gnn import GNNMolecularPredictor

__all__ = [
    # 'BaseMolecularPredictor',
    'GREAMolecularPredictor',
    'GNNMolecularPredictor',
]