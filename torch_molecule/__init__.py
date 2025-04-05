__version__ = "0.1.0.dev0"

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
from .predictor.smiles_transformer import SMILESTransformerMolecularPredictor
"""
encoder module
"""
from .encoder.supervised import SupervisedMolecularEncoder
from .encoder.attrmask import AttrMaskMolecularEncoder
from .encoder.contextpred import ContextPredMolecularEncoder
from .encoder.edgepred import EdgePredMolecularEncoder
from .encoder.moama import MoamaMolecularEncoder
from .encoder.infograph import InfoGraphMolecularEncoder
from .encoder.pretrained import HFPretrainedMolecularEncoder
"""
generator module
"""
from .generator.graph_dit import GraphDITMolecularGenerator
from .generator.graph_ga import GraphGAMolecularGenerator
from .generator.digress import DigressMolecularGenerator
from .generator.molgpt import MolGPTMolecularGenerator

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
    'SMILESTransformerMolecularPredictor',
    # encoders
    'SupervisedMolecularEncoder',
    'AttrMaskMolecularEncoder',
    'ContextPredMolecularEncoder',
    'EdgePredMolecularEncoder',
    'MoamaMolecularEncoder',
    'InfoGraphMolecularEncoder',
    'HFPretrainedMolecularEncoder',
    # generators
    'GraphDITMolecularGenerator',
    'GraphGAMolecularGenerator',
    'DigressMolecularGenerator',
    'MolGPTMolecularGenerator',
]