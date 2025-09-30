import numpy as np
import warnings
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, List, Callable, Literal

import torch
from torch_geometric.loader import DataLoader

from .model import GREA
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

class GREAMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements GREA model from the paper "Graph Rationalization with Environment-based Augmentations".

    During model training, it learns the rationales (explainable subgraphs). Use them for data augmentation.
    During prediction, the model uses the rationales to make predictions and the rationales themselves can also explain the predictions.

    References
    ----------
    - Graph Rationalization with Environment-based Augmentations.
      https://dl.acm.org/doi/10.1145/3534678.3539347
    - Code: https://github.com/liugangcode/GREA

    Parameters
    ----------
    gamma : float, default=0.4
        GREA-specific parameter that penalize the size of the rationales (ratio between the number of nodes in the rationales and the number of nodes in the original graph).
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
    verbose : bool, default=False
        Whether to print progress information during training.
    device : torch.device or str, optional
        Device to use for computation.
    model_name : str, default="GREAMolecularPredictor"
        Name of the model.
    """
    def __init__(
        self,
        # GREA-specific parameters
        gamma: float = 0.4,
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
        model_name: str = "GREAMolecularPredictor",
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
        
        # GREA-specific parameters
        self.gamma = gamma
        self.model_class = GREA

    @staticmethod
    def _get_param_names() -> List[str]:
        return ["gamma"] + GNNMolecularPredictor._get_param_names()

    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["gamma"] = ParameterSpec(ParameterType.FLOAT, (0.1, 0.9))
        return search_space

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        base_params = super()._get_model_params(checkpoint)
        if checkpoint and "hyperparameters" in checkpoint:
            base_params["gamma"] = checkpoint["hyperparameters"].get("gamma", self.gamma)
        else:
            base_params["gamma"] = self.gamma
        base_params.pop("graph_pooling", None)
        return base_params

    def predict(self, X: List[str]) -> Dict[str, Union[np.ndarray, List[List]]]:
        """Make predictions using the fitted model.

        Parameters
        ----------
        X : List[str]
            List of SMILES strings to make predictions for

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
                - 'prediction': Model predictions (shape: [n_samples, n_tasks])
                - 'variance': Prediction variances (shape: [n_samples, n_tasks])
                - 'node_importance': A nested list where the outer list has length n_samples and each inner list has length n_nodes for that molecule
        """
        self._check_is_fitted()

        # Convert to PyTorch Geometric format and create loader
        X, _ = self._validate_inputs(X)
        dataset = self._convert_to_pytorch_data(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Make predictions
        self.model = self.model.to(self.device)
        self.model.eval()
        predictions = []
        variances = []
        node_scores = []
        with torch.no_grad():
            for batch in tqdm(loader, disable=self.verbose != "progress_bar"):
                batch = batch.to(self.device)
                out = self.model(batch)
                predictions.append(out["prediction"].cpu().numpy())
                variances.append(out["variance"].cpu().numpy())
                node_scores.extend(out["score"])

        if predictions and variances:
            return {
                "prediction": np.concatenate(predictions, axis=0),
                "variance": np.concatenate(variances, axis=0),
                "node_importance": node_scores,
            }
        else:
            warnings.warn(
                "No valid predictions could be made from the input data. Returning empty results."
            )
            return {"prediction": np.array([]), "variance": np.array([]), "node_importance": np.array([])}