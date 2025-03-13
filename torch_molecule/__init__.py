__version__ = "1.0.0.dev0"

"""
torch_molecule.predictor module
"""
from torch_molecule.predictor.grea import GREAMolecularPredictor
from torch_molecule.predictor.gnn import GNNMolecularPredictor
from torch_molecule.predictor.sgir import SGIRMolecularPredictor
from torch_molecule.predictor.irm import IRMMolecularPredictor
from torch_molecule.predictor.ssr import SSRMolecularPredictor
from torch_molecule.predictor.rpgnn import RPGNNMolecularPredictor
from torch_molecule.predictor.lstm import LSTMMolecularPredictor
"""
torch_molecule.encoder module
"""
from torch_molecule.encoder.supervised import SupervisedMolecularEncoder
from torch_molecule.encoder.attrmask import AttrMaskMolecularEncoder
from torch_molecule.encoder.contextpred import ContextPredMolecularEncoder
from torch_molecule.encoder.edgepred import EdgePredMolecularEncoder
from torch_molecule.encoder.moama import MoamaMolecularEncoder

"""
torch_molecule.generator module
"""
from torch_molecule.generator.graph_dit import GraphDITMolecularGenerator
from torch_molecule.generator.graphga import GraphGAMolecularGenerator

__all__ = [
    # 'BaseMolecularPredictor',
    # predictors
    'SGIRMolecularPredictor',
    'GREAMolecularPredictor',
    'GNNMolecularPredictor',
    'IRMMolecularPredictor',
    'SSRMolecularPredictor',
    'RPGNNMolecularPredictor',
    'LSTMMolecularPredictor',
    # encoders
    'SupervisedMolecularEncoder',
    'AttrMaskMolecularEncoder',
    'ContextPredMolecularEncoder',
    'EdgePredMolecularEncoder',
    'MoamaMolecularEncoder',
    # generators
    'GraphDITMolecularGenerator',
    'GraphGAMolecularGenerator',
]