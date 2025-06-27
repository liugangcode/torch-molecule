__version__ = "0.1.3"

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
from .predictor.bfgnn import BFGNNMolecularPredictor
from .predictor.grin import GRINMolecularPredictor
from .predictor.lstm import LSTMMolecularPredictor
from .predictor.smiles_transformer import SMILESTransformerMolecularPredictor
"""
encoder module
"""
from .encoder.attrmask import AttrMaskMolecularEncoder
from .encoder.moama import MoamaMolecularEncoder
from .encoder.graphmae import GraphMAEMolecularEncoder
from .encoder.supervised import SupervisedMolecularEncoder
from .encoder.contextpred import ContextPredMolecularEncoder
from .encoder.edgepred import EdgePredMolecularEncoder
from .encoder.infograph import InfoGraphMolecularEncoder
from .encoder.pretrained import HFPretrainedMolecularEncoder
"""
generator module
"""
from .generator.graph_dit import GraphDITMolecularGenerator
from .generator.digress import DigressMolecularGenerator
from .generator.gdss import GDSSMolecularGenerator
from .generator.graph_ga import GraphGAMolecularGenerator
from .generator.jtvae import JTVAEMolecularGenerator
from .generator.lstm import LSTMMolecularGenerator
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
    'BFGNNMolecularPredictor',
    'GRINMolecularPredictor',
    'LSTMMolecularPredictor',
    'SMILESTransformerMolecularPredictor',
    # encoders
    'SupervisedMolecularEncoder',
    'AttrMaskMolecularEncoder',
    'MoamaMolecularEncoder',
    'GraphMAEMolecularEncoder',
    'ContextPredMolecularEncoder',
    'EdgePredMolecularEncoder',
    'InfoGraphMolecularEncoder',
    'HFPretrainedMolecularEncoder',
    # generators
    'GraphDITMolecularGenerator',
    'DigressMolecularGenerator',
    'GDSSMolecularGenerator',
    'GraphGAMolecularGenerator',
    'JTVAEMolecularGenerator',
    'MolGPTMolecularGenerator',
    'LSTMMolecularGenerator',
]