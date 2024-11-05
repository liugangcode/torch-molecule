import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable

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
    """

    def __init__(
        self,
        gamma: float = 0.4,
        # model parameters
        num_tasks: int = 1,
        task_type: str = "classification",
        num_layer: int = 5,
        emb_dim: int = 300,
        gnn_type: str = "gin-virtual",
        drop_ratio: float = 0.5,
        norm_layer: str = "batch_norm",
        graph_pooling: str = "max",
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
            num_tasks=num_tasks,
            task_type=task_type,
            num_layer=num_layer,
            emb_dim=emb_dim,
            gnn_type=gnn_type,
            drop_ratio=drop_ratio,
            norm_layer=norm_layer,
            graph_pooling=graph_pooling,
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

        # Initialize GREA model
        # self._initialize_model(GREA, self.device)
        # self.model = GREA(
        #     num_tasks=self.num_tasks,
        #     num_layer=self.num_layer,
        #     gamma=self.gamma,
        #     emb_dim=self.emb_dim,
        #     gnn_type=self.gnn_type,
        #     drop_ratio=self.drop_ratio,
        #     norm_layer=self.norm_layer,
        # ).to(self.device)

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        List[str]
            List of parameter names that can be used for model configuration.
        """
        return [
            "gamma",
            # Model hyperparameters
            "num_tasks",
            "task_type",
            "num_layer",
            "emb_dim",
            "gnn_type",
            "drop_ratio",
            "norm_layer",
            "graph_pooling",
            # Training parameters
            "batch_size",
            "epochs",
            "learning_rate",
            "grad_clip_value",
            "weight_decay",
            "patience",
            "loss_criterion",
            # evaluation parameters
            "evaluate_name",
            "evaluate_criterion",
            "evaluate_higher_better",
            # Scheduler parameters
            "use_lr_scheduler",
            "scheduler_factor",
            "scheduler_patience",
            # others
            "device",
            "fitting_epoch",
            "fitting_loss",
            "verbose"
        ]

    def _get_default_search_space(self):
        """Get the default hyperparameter search space.
        
        Returns
        -------
        Dict[str, ParameterSpec]
            Dictionary mapping parameter names to their search space specifications
        """
        default_parameters = DEFAULT_GNN_SEARCH_SPACES.copy()
        
        # Add gamma parameter to the search space
        default_parameters.update({
            "gamma": ParameterSpec(
                ParameterType.FLOAT,
                (0.1, 0.9)
            )
        })
        return default_parameters

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        """Get model parameters either from checkpoint or current instance.
        
        Parameters
        ----------
        checkpoint : Optional[Dict]
            Checkpoint containing model hyperparameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters
            
        Raises
        ------
        ValueError
            If checkpoint contains invalid parameters
        """
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
                
            hyperparameters = checkpoint["hyperparameters"]
            
            # Define required parameters
            required_params = {
                "gamma", "num_tasks", "num_layer", "emb_dim", "gnn_type",
                "drop_ratio", "norm_layer"
            }
            
            # Validate parameters
            invalid_params = set(hyperparameters.keys()) - required_params
            # if invalid_params:
            #     raise ValueError(f"Invalid parameters in checkpoint: {invalid_params}")
            
            # Get parameters with fallback to instance values
            return {
                "gamma": hyperparameters['gamma'],
                "num_tasks": hyperparameters['num_tasks'],
                "num_layer": hyperparameters['num_layer'],
                "emb_dim": hyperparameters['emb_dim'],
                "gnn_type": hyperparameters['gnn_type'],
                "drop_ratio": hyperparameters['drop_ratio'],
                "norm_layer": hyperparameters['norm_layer'],
            }
        else:
            # Use current instance parameters
            return {
                "gamma": self.gamma,
                "num_tasks": self.num_tasks,
                "num_layer": self.num_layer,
                "emb_dim": self.emb_dim,
                "gnn_type": self.gnn_type,
                "drop_ratio": self.drop_ratio,
                "norm_layer": self.norm_layer,
            }
        
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
                - 'confidence': Prediction confidences (shape: [n_samples, n_tasks])
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

    def save_model(self, path: str) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : str
            Path where to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")

        if not path.endswith((".pt", ".pth")):
            raise ValueError("Save path should end with '.pt' or '.pth'")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        model_name = os.path.splitext(os.path.basename(path))[0]
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "hyperparameters": self.get_params(),
            "model_name": model_name,
            "date_saved": datetime.datetime.now().isoformat(),
            "version": getattr(self, "__version__", "1.0.0"),
        }
        torch.save(save_dict, path)