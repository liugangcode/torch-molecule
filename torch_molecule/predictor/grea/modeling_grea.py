import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal

import torch
from torch_geometric.loader import DataLoader

from .architecture import GREA
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    DEFAULT_GNN_SEARCH_SPACES,
    ParameterSpec,
    ParameterType,
)
class GREAMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements a GNN model based on Graph Rationalization
    for molecular property prediction tasks.
    Paper: Graph Rationalization with Environment-based Augmentations (https://dl.acm.org/doi/10.1145/3534678.3539347)
    Reference Code: https://github.com/liugangcode/GREA 
    """
    def __init__(
        self,
        gamma: float = 0.4,
        # model parameters
        num_task: int = 1,
        task_type: str = "classification",
        num_layer: int = 5,
        hidden_size: int = 300,
        gnn_type: str = "gin-virtual",
        drop_ratio: float = 0.5,
        norm_layer: str = "batch_norm",
        graph_pooling: str = "sum",
        # augmented features
        augmented_feature: Optional[list[Literal["morgan", "maccs"]]] = ["morgan", "maccs"],
        # training parameters
        batch_size: int = 128,
        epochs: int = 500,
        loss_criterion: Optional[Callable] = None,
        evaluate_criterion: Optional[Union[str, Callable]] = None,
        evaluate_higher_better: Optional[bool] = None,
        learning_rate: float = 0.001,
        grad_clip_value: Optional[float] = None,
        weight_decay: float = 0.0,
        patience: int = 50,
        # scheduler
        use_lr_scheduler: bool = True,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        # others
        device: Optional[str] = None,
        verbose: bool = False,
        model_name: str = "GREAMolecularPredictor",
    ):
        # Call parent class's __init__ with all the common parameters
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
            loss_criterion=loss_criterion,
            evaluate_criterion=evaluate_criterion,
            evaluate_higher_better=evaluate_higher_better,
            learning_rate=learning_rate,
            grad_clip_value=grad_clip_value,
            weight_decay=weight_decay,
            patience=patience,
            use_lr_scheduler=use_lr_scheduler,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            device=device,
            verbose=verbose,
            model_name=model_name,
        )
        # GREA-specific parameter
        self.gamma = gamma
        self.model_class = GREA

    @staticmethod
    def _get_param_names() -> List[str]:
        return ["gamma"] + GNNMolecularPredictor._get_param_names()

    def _get_default_search_space(self):
        search_space = super()._get_default_search_space()
        search_space["gamma"] = ParameterSpec(ParameterType.FLOAT, (0.1, 0.9))
        return search_space

    # def _get_default_search_space(self):
    #     """Get the default hyperparameter search space.
        
    #     Returns
    #     -------
    #     Dict[str, ParameterSpec]
    #         Dictionary mapping parameter names to their search space specifications
    #     """
    #     default_parameters = DEFAULT_GNN_SEARCH_SPACES.copy()
        
    #     # Add gamma parameter to the search space
    #     default_parameters.update({
    #         "gamma": ParameterSpec(
    #             ParameterType.FLOAT,
    #             (0.1, 0.9)
    #         )
    #     })
    #     return default_parameters

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        base_params = super()._get_model_params(checkpoint)
        if checkpoint and "hyperparameters" in checkpoint:
            base_params["gamma"] = checkpoint["hyperparameters"].get("gamma", self.gamma)
        else:
            base_params["gamma"] = self.gamma
        base_params.pop("graph_pooling", None)
        return base_params
        # if checkpoint is not None:
        #     if "hyperparameters" not in checkpoint:
        #         raise ValueError("Checkpoint missing 'hyperparameters' key")
                
        #     hyperparameters = checkpoint["hyperparameters"]
            
        #     # Define required parameters
        #     required_params = {
        #         "gamma", "num_task", "num_layer", "hidden_size", "gnn_type",
        #         "drop_ratio", "norm_layer", "augmented_feature"
        #     }
            
        #     # Validate parameters
        #     invalid_params = set(hyperparameters.keys()) - required_params
        #     # if invalid_params:
        #     #     raise ValueError(f"Invalid parameters in checkpoint: {invalid_params}")
            
        #     # Get parameters with fallback to instance values
        #     return {
        #         "gamma": hyperparameters['gamma'],
        #         "num_task": hyperparameters['num_task'],
        #         "num_layer": hyperparameters['num_layer'],
        #         "hidden_size": hyperparameters['hidden_size'],
        #         "gnn_type": hyperparameters['gnn_type'],
        #         "drop_ratio": hyperparameters['drop_ratio'],
        #         "norm_layer": hyperparameters['norm_layer'],
        #         "augmented_feature": hyperparameters.get("augmented_feature", self.augmented_feature)
        #     }
        # else:
        #     # Use current instance parameters
        #     return {
        #         "gamma": self.gamma,
        #         "num_task": self.num_task,
        #         "num_layer": self.num_layer,
        #         "hidden_size": self.hidden_size,
        #         "gnn_type": self.gnn_type,
        #         "drop_ratio": self.drop_ratio,
        #         "norm_layer": self.norm_layer,
        #         "augmented_feature": self.augmented_feature
        #     }
        
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
            for batch in tqdm(loader, disable=not self.verbose):
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
            return {"prediction": np.array([]), "variance": np.array([])}