from tqdm import tqdm
from typing import Optional, Dict, Any, List, Callable, Union, Literal

import torch

from .model import BFGNN
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

class BFGNNMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements algorithm alignment of Bellman-Ford algorithm with GNN.

    References
    ----------
    - Graph neural networks extrapolate out-of-distribution for shortest paths.
      https://arxiv.org/abs/2503.19173

    Parameters
    ----------
    l1_penalty : float, default=1e-3
        L1 regularization penalty strength for feature selection.
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
    """
    
    def __init__(
        self,
        # BFGNN-specific parameters
        l1_penalty: float = 1e-3,
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
        device: Optional[torch.device | str] = None,
        model_name: str = "BFGNNMolecularPredictor",
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
        
        # BFGNN-specific parameters
        self.l1_penalty = l1_penalty
        self.model_class = BFGNN

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

    def _train_epoch(self, train_loader, optimizer, epoch, global_pbar=None):
        self.model.train()
        losses = []

        for step, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()

            loss = self.model.compute_loss(batch, self.loss_criterion, self.l1_penalty)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()

            if global_pbar is not None:
                global_pbar.update(1)
                global_pbar.set_postfix({
                    "Epoch": f"{epoch+1}/{self.epochs}",
                    "Batch": f"{step+1}/{len(train_loader)}",
                    "Loss": f"{loss.item():.4f}"
                })
            losses.append(loss.item())

        return losses