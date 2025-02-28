__version__ = "1.0.0.dev0"

"""
torch_molecule.predictor module
"""
# from torch_molecule.base import BaseMolecularPredictor
from torch_molecule.predictor.grea import GREAMolecularPredictor
from torch_molecule.predictor.gnn import GNNMolecularPredictor
from torch_molecule.predictor.sgir import SGIRMolecularPredictor
from torch_molecule.predictor.irm import IRMMolecularPredictor

"""
torch_molecule.encoder module
"""
from torch_molecule.encoder.supervised import SupervisedMolecularEncoder

__all__ = [
    # 'BaseMolecularPredictor',
    # predictors
    'SGIRMolecularPredictor',
    'GREAMolecularPredictor',
    'GNNMolecularPredictor',
    'IRMMolecularPredictor',
    # encoders
    'SupervisedMolecularEncoder',
]