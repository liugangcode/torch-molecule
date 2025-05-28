import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, List, Type
from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

from .model import GRIN
from ...utils import graph_from_smiles
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

@dataclass
class GRINMolecularPredictor(GNNMolecularPredictor):
    """
    This predictor implements GRIN for Max Spanning Tree algorithm aligned GNN.

    The full name of GRIN is Graph Invariant Representation Learning.

    References
    ----------
    - Learning Repetition-Invariant Representations for Polymer Informatics.
      https://arxiv.org/pdf/2505.10726

    :param l1_penalty: Weight for the L1 penalty
    :type l1_penalty: float, default=1e-3
    :param epochs_to_penalize: Number of epochs to train before starting L1 penalty
    :type epochs_to_penalize: int, default=100
    """

    l1_penalty: float = 1e-3
    epochs_to_penalize: int = 100

    # Other Non-init fields
    model_name: str = "GRINMolecularPredictor"
    model_class: Type[GRIN] = field(default=GRIN, init=False)
    
    def __post_init__(self):
        super().__post_init__()
    
    @staticmethod
    def _get_param_names() -> List[str]:
        return GNNMolecularPredictor._get_param_names() + [
            "l1_penalty",
            "epochs_to_penalize"
        ]
    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["l1_penalty"] = ParameterSpec(ParameterType.LOG_FLOAT, (1e-6, 1))
        search_space["epochs_to_penalize"] = ParameterSpec(ParameterType.INTEGER, (0, 100))
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
            if epoch >= self.epochs_to_penalize:
                l1_penalty = min(epoch - self.epochs_to_penalize, 1) * self.l1_penalty
            else:
                l1_penalty = 0
            loss = self.model.compute_loss(batch, self.loss_criterion, l1_penalty)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()

            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Total Loss": f"{loss.item():.4f}"})
            losses.append(loss.item())

        return losses