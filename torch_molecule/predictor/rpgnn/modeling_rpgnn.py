import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal, Type
from dataclasses import dataclass, field

import torch
from torch_geometric.loader import DataLoader

from .model import RPGNN
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

@dataclass
class RPGNNMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements a GNN model based on Relational pooling.

    The full name of RPGNN is Relational Pooling for Graph Representations.

    References
    ----------
    - Relational Pooling for Graph Representations.
      https://arxiv.org/abs/1903.02541
    - Reference Code: https://github.com/PurdueMINDS/RelationalPooling/tree/master?tab=readme-ov-file

    Parameters
    ----------
    num_perm : int, default=3
        Number of random permutations to use for relational pooling. 
    fixed_size : int, default=10 
        Maximum number of nodes to consider in the graph. 
    num_node_feature : int, default=9
        Dimension of the input node features. This should match the number of atomic features used to represent
        each node in the molecular graph (e.g., atomic number, degree, hybridization, etc.).
    """
    
    # RPGNN-specific parameter
    num_perm: int = 3
    fixed_size: int = 10
    num_node_feature: int = 9
    # Override parent defaults
    model_name: str = "RPGNNMolecularPredictor"
    model_class: Type[RPGNN] = field(default=RPGNN, init=False)
    
    def __post_init__(self):
        super().__post_init__()

    @staticmethod
    def _get_param_names() -> List[str]:
        return ["num_perm", "fixed_size", "num_node_feature"] + GNNMolecularPredictor._get_param_names()

    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["num_perm"] = ParameterSpec(ParameterType.INTEGER, (1, 10))
        search_space["fixed_size"] = ParameterSpec(ParameterType.INTEGER, (1, 10))
        return search_space

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        base_params = super()._get_model_params(checkpoint)
        if checkpoint and "hyperparameters" in checkpoint:
            base_params["num_perm"] = checkpoint["hyperparameters"]["num_perm"]
            base_params["fixed_size"] = checkpoint["hyperparameters"]["fixed_size"]
            base_params["num_node_feature"] = checkpoint["hyperparameters"]["num_node_feature"]
        else:
            base_params["num_perm"] = self.num_perm
            base_params["fixed_size"] = self.fixed_size
            base_params["num_node_feature"] = self.num_node_feature
        base_params.pop("graph_pooling", None)
        return base_params

    