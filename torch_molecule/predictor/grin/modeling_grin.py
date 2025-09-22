from typing import Optional, Union, Dict, Any, List, Callable, Literal

import torch

from .model import GRIN
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

class GRINMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements GRIN for Max Spanning Tree algorithm aligned GNN.

    The full name of GRIN is Graph Invariant Representation Learning.

    References
    ----------
    - Learning Repetition-Invariant Representations for Polymer Informatics.
      https://arxiv.org/pdf/2505.10726

    Parameters
    ----------
    l1_penalty : float, default=1e-3
        Weight for the L1 penalty.
    epochs_to_penalize : int, default=100
        Number of epochs to train before starting L1 penalty.
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
        Device to use for computation.
    model_name : str, default="GRINMolecularPredictor"
        Name of the model.
    """
    def __init__(
        self,
        # GRIN-specific parameters
        l1_penalty: float = 1e-3,
        epochs_to_penalize: int = 100,
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
        model_name: str = "GRINMolecularPredictor"
    ):
        super().__init__(
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
            verbose=verbose,
            device=device,
            model_name=model_name,
        )
        
        # GRIN-specific parameters
        self.l1_penalty = l1_penalty
        self.epochs_to_penalize = epochs_to_penalize
        self.model_class = GRIN

    
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

    def _train_epoch(self, train_loader, optimizer, epoch, global_pbar=None):
        self.model.train()
        losses = []

        for batch_idx, batch in enumerate(train_loader):
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

            if global_pbar is not None:
                global_pbar.update(1)
                global_pbar.set_postfix({
                    "Epoch": f"{epoch+1}/{self.epochs}",
                    "Batch": f"{batch_idx+1}/{len(train_loader)}",
                    "Loss": f"{loss.item():.4f}"
                })
            losses.append(loss.item())

        return losses