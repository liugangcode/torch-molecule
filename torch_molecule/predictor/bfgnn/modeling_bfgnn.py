import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, List, Type
from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

from .model import BFGNN
from ...utils import graph_from_smiles
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

@dataclass
class BFGNNMolecularPredictor(GNNMolecularPredictor):
    """
    This predictor implements algorithm alignment of Bellman-Ford algorithm with GNN.

    References
    ----------
    - Graph neural networks extrapolate out-of-distribution for shortest paths.
      https://arxiv.org/abs/2503.19173

    :param l1_penalty: Weight for the L1 penalty
    :type l1_penalty: float, default=1e-3
    """


    l1_penalty: float = 1e-3
    # Other Non-init fields
    model_name: str = "BFGNNMolecularPredictor"
    model_class: Type[BFGNN] = field(default=BFGNN, init=False)
    
    def __post_init__(self):
        super().__post_init__()
    
    @staticmethod
    def _get_param_names() -> List[str]:
        return GNNMolecularPredictor._get_param_names() + [
            "l1_penalty",
        ]
    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["l1_penalty"] = ParameterSpec(ParameterType.LOG_FLOAT, (1e-6, 1))
        return search_space

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        base_params = super()._get_model_params(checkpoint)
        return base_params

    def _train_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        losses = []

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )

        for step, batch in enumerate(iterator):
            batch = batch.to(self.device)
            optimizer.zero_grad()

            loss = self.model.compute_loss(batch, self.loss_criterion, self.l1_penalty)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()

            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Total Loss": f"{loss.item():.4f}"})
            losses.append(loss.item())

        return losses