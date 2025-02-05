__version__ = "1.0.0.dev0"

"""
torch_molecule.predictor module
Provides molecular property prediction functionality
"""

# from torch_molecule.base import BaseMolecularPredictor
from torch_molecule.predictor.grea import GREAMolecularPredictor
from torch_molecule.predictor.gnn import GNNMolecularPredictor
from torch_molecule.predictor.sgir import SGIRMolecularPredictor

from torch_molecule.encoder.supervised import SupervisedMolecularEncoder

__all__ = [
    # 'BaseMolecularPredictor',
    'SGIRMolecularPredictor',
    'GREAMolecularPredictor',
    'GNNMolecularPredictor',
    'SupervisedMolecularEncoder',
]