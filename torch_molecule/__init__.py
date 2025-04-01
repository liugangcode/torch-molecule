__version__ = "1.0.0.dev0"

"""
predictor module
"""
from .predictor.grea import GREAMolecularPredictor
from .predictor.gnn import GNNMolecularPredictor
from .predictor.sgir import SGIRMolecularPredictor
from .predictor.irm import IRMMolecularPredictor
from .predictor.ssr import SSRMolecularPredictor
from .predictor.dir import DIRMolecularPredictor
from .predictor.rpgnn import RPGNNMolecularPredictor
from .predictor.lstm import LSTMMolecularPredictor
from .predictor.transformer import TransformerMolecularPredictor
"""
encoder module
"""
from .encoder.supervised import SupervisedMolecularEncoder
from .encoder.attrmask import AttrMaskMolecularEncoder
from .encoder.contextpred import ContextPredMolecularEncoder
from .encoder.edgepred import EdgePredMolecularEncoder
from .encoder.moama import MoamaMolecularEncoder

"""
generator module
"""
from .generator.graph_dit import GraphDITMolecularGenerator
from .generator.graph_ga import GraphGAMolecularGenerator
from .generator.digress import DigressMolecularGenerator
__all__ = [
    # 'BaseMolecularPredictor',
    # predictors
    'SGIRMolecularPredictor',
    'GREAMolecularPredictor',
    'GNNMolecularPredictor',
    'IRMMolecularPredictor',
    'SSRMolecularPredictor',
    'DIRMolecularPredictor',
    'RPGNNMolecularPredictor',
    'LSTMMolecularPredictor',
    'TransformerMolecularPredictor',
    # encoders
    'SupervisedMolecularEncoder',
    'AttrMaskMolecularEncoder',
    'ContextPredMolecularEncoder',
    'EdgePredMolecularEncoder',
    'MoamaMolecularEncoder',
    # generators
    'GraphDITMolecularGenerator',
    'GraphGAMolecularGenerator',
    'DigressMolecularGenerator',
]