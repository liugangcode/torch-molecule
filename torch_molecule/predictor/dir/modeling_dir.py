import numpy as np
import warnings
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal

import torch
from torch_geometric.loader import DataLoader

from .model import DIR
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

class DIRMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements the DIR for molecular property prediction tasks.

    The full name of DIR is Discovering Invariant Rationales.

    References
    ----------
    - Discovering Invariant Rationales for Graph Neural Networks.
      https://openreview.net/forum?id=hGXij5rfiHw
    - Code: https://github.com/Wuyxin/DIR-GNN
        
    Parameters
    ----------
    causal_ratio : float, default=0.8
        The ratio of causal edges to keep during training. A higher ratio means more edges
        are considered causal/important for the prediction. This controls the sparsity of
        the learned rationales.
        
    lw_invariant : float, default=1e-4
        The weight of the invariance loss term. This loss encourages the model to learn
        rationales that are invariant across different environments/perturbations. A higher
        value puts more emphasis on learning invariant features.
        
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
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    weight_decay : float, default=0.0
        L2 regularization strength.
    grad_clip_value : float, optional
        Maximum norm of gradients for gradient clipping.
    patience : int, default=50
        Number of epochs to wait for improvement before early stopping.
    use_lr_scheduler : bool, default=False
        Whether to use learning rate scheduler.
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate when plateau is reached.
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced.
    loss_criterion : callable, optional
        Loss function for training.
    evaluate_criterion : str or callable, optional
        Metric for model evaluation.
    evaluate_higher_better : bool, optional
        Whether higher values of the evaluation metric are better.
    verbose : str, default="none"
        Whether to display progress info. Options are: "none", "progress_bar", "print_statement". If any other, "none" is automatically chosen.
    device : torch.device or str, optional
        Device to use for computation.
    model_name : str, default="DIRMolecularPredictor"
        Name of the model.
    """
    def __init__(
        self,
        # DIR-specific parameters
        causal_ratio: float = 0.8,
        lw_invariant: float = 1e-4,
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
        model_name: str = "DIRMolecularPredictor",
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
        
        # DIR-specific parameters
        self.causal_ratio = causal_ratio
        self.lw_invariant = lw_invariant
        self.model_class = DIR

    @staticmethod
    def _get_param_names() -> List[str]:
        return ["causal_ratio", "lw_invariant"] + GNNMolecularPredictor._get_param_names()

    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["causal_ratio"] = ParameterSpec(ParameterType.FLOAT, (0.1, 0.9))
        search_space["lw_invariant"] = ParameterSpec(ParameterType.FLOAT, (1e-5, 1e-2))
        return search_space

    def _setup_optimizers(self) -> Tuple[Dict[str, torch.optim.Optimizer], Optional[Any]]:
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        conf_optimizer = torch.optim.Adam(self.model.conf_lin.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.grad_clip_value is not None:
            for group in model_optimizer.param_groups:
                group.setdefault("max_norm", self.grad_clip_value)
                group.setdefault("norm_type", 2.0)

        scheduler = None
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                model_optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=1e-6,
                cooldown=0,
                eps=1e-8,
            )
        optimizer = {"model": model_optimizer, "conf": conf_optimizer}

        return optimizer, scheduler
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        base_params = super()._get_model_params(checkpoint)
        if checkpoint and "hyperparameters" in checkpoint:
            base_params["causal_ratio"] = checkpoint["hyperparameters"]["causal_ratio"]
        else:
            base_params["causal_ratio"] = self.causal_ratio
        return base_params

    def _train_epoch(self, train_loader, optimizer, epoch, global_pbar=None):
        self.model.train()
        losses = []

        alpha_prime = self.lw_invariant * (epoch ** 1.6)
        conf_opt = optimizer["conf"]
        model_optimizer = optimizer["model"]

        for batch_idx, batch in enumerate(train_loader):

            batch = batch.to(self.device)

            # Forward pass and loss computation
            causal_loss, conf_loss, env_loss = self.model.compute_loss(batch, self.loss_criterion, alpha_prime)

            conf_opt.zero_grad()
            conf_loss.backward()
            conf_opt.step()

            model_optimizer.zero_grad()
            (causal_loss + env_loss).backward()
            model_optimizer.step()

            loss = causal_loss + env_loss + conf_loss
            losses.append(loss.item())

            # Update progress bar if using tqdm
            if global_pbar is not None:
                global_pbar.update(1)
                global_pbar.set_postfix({
                    "Epoch": f"{epoch+1}/{self.epochs}",
                    "Batch": f"{batch_idx+1}/{len(train_loader)}",
                    "Loss": f"{loss.item():.4f}"
                })

        return losses
    
    def predict(self, X: List[str]) -> Dict[str, Union[np.ndarray, List[List]]]:
        self._check_is_fitted()

        # Convert to PyTorch Geometric format and create loader
        X, _ = self._validate_inputs(X)
        dataset = self._convert_to_pytorch_data(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Make predictions
        self.model = self.model.to(self.device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, disable=self.verbose != "progress_bar"):
                batch = batch.to(self.device)
                out = self.model(batch)
                predictions.append(out["prediction"].cpu().numpy())

        if predictions:
            return {
                "prediction": np.concatenate(predictions, axis=0),
            }
        else:
            warnings.warn(
                "No valid predictions could be made from the input data. Returning empty results."
            )
            return {"prediction": np.array([])}


