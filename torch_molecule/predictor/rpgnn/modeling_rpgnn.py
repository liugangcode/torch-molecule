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
    num_task : int, default=1
        Number of prediction tasks.
    task_type : str, default="regression"
        Type of prediction task, either "regression" or "classification".
    num_layer : int, default=5
        Number of GNN layers.
    hidden_size : int, default=300
        Dimension of hidden node features.
    gnn_type : str, default="gin-virtual"
        Type of GNN architecture to use. One of ["gin-virtual", "gcn-virtual", "gin", "gcn"].
    drop_ratio : float, default=0.5
        Dropout probability.
    norm_layer : str, default="batch_norm"
        Type of normalization layer to use. One of ["batch_norm", "layer_norm", "instance_norm", "graph_norm", "size_norm", "pair_norm"].
    graph_pooling : str, default="sum"
        Method for aggregating node features to graph-level representations. One of ["sum", "mean", "max"].
    augmented_feature : list or None, default=None
        Additional molecular fingerprints to use as features. It will be concatenated with the graph representation after pooling.
        Examples like ["morgan", "maccs"] or None.
    batch_size : int, default=128
        Number of samples per batch for training.
    epochs : int, default=500
        Maximum number of training epochs.
    loss_criterion : callable, optional
        Loss function for training.
    evaluate_criterion : str or callable, optional
        Metric for model evaluation.
    evaluate_higher_better : bool, optional
        Whether higher values of the evaluation metric are better.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    grad_clip_value : float, optional
        Maximum norm of gradients for gradient clipping.
    weight_decay : float, default=0.0
        L2 regularization strength.
    patience : int, default=50
        Number of epochs to wait for improvement before early stopping.
    use_lr_scheduler : bool, default=False
        Whether to use learning rate scheduler.
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate when plateau is reached.
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced.
    verbose : str, default="none"
        Whether to display progress info. Options are: "none", "progress_bar", "print_statement". If any other, "none" is automatically chosen.
    device : torch.device or str, optional
        Device to run the model on. If None, will auto-detect GPU or use CPU.
    model_name : str, default="RPGNNMolecularPredictor"
        Name identifier for the model.
    num_perm : int, default=3
        Number of random permutations to use for relational pooling. 
    fixed_size : int, default=10 
        Maximum number of nodes to consider in the graph. 
    num_node_feature : int, default=9
        Dimension of the input node features. This should match the number of atomic features used to represent
        each node in the molecular graph (e.g., atomic number, degree, hybridization, etc.).
    """
    def __init__(
        self,
        # RPGNN-specific parameters
        num_perm: int = 3,
        fixed_size: int = 10,
        num_node_feature: int = 9,
        # Core model parameters
        num_task: int = 1,
        task_type: str = "regression",
        # GNN architecture parameters
        num_layer: int = 5,
        hidden_size: int = 300,
        gnn_type: str = "gin-virtual",
        drop_ratio: float = 0.5,
        norm_layer: str = "batch_norm",
        graph_pooling: str = "sum",
        augmented_feature: Optional[list[Literal["morgan", "maccs"]]] = None,
        # Training parameters
        batch_size: int = 128,
        epochs: int = 500,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        grad_clip_value: Optional[float] = None,
        patience: int = 50,
        # Learning rate scheduler parameters
        use_lr_scheduler: bool = False,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        # Loss and evaluation parameters
        loss_criterion: Optional[Callable] = None,
        evaluate_criterion: Optional[Union[str, Callable]] = None,
        evaluate_higher_better: Optional[bool] = None,
        # General parameters
        verbose: str = "none",
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "RPGNNMolecularPredictor",
    ):
        super().__init__(
            device=device,
            model_name=model_name,
            num_task=num_task,
            task_type=task_type,
            num_layer=num_layer,
            hidden_size=hidden_size,
            gnn_type=gnn_type,
            drop_ratio=drop_ratio,
            norm_layer=norm_layer,
            graph_pooling=graph_pooling,
            augmented_feature=augmented_feature,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_value=grad_clip_value,
            patience=patience,
            use_lr_scheduler=use_lr_scheduler,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            loss_criterion=loss_criterion,
            evaluate_criterion=evaluate_criterion,
            evaluate_higher_better=evaluate_higher_better,
            verbose=verbose
        )
        self.num_perm = num_perm
        self.fixed_size = fixed_size
        self.num_node_feature = num_node_feature
        self.model_class = RPGNN

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