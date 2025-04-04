import numpy as np
import warnings
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, List, Type
from dataclasses import dataclass, field

import torch
from torch_geometric.loader import DataLoader

from .model import GREA
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

@dataclass
class GREAMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements a Graph Rationalization model called GREA.

    The full name of GREA is Graph Rationalization with Environment-based Augmentations. During model training, it learns the rationales (explainable subgraphs) and use them for molecular property prediction tasks.

    References
    ----------
    - Graph Rationalization with Environment-based Augmentations.
      https://dl.acm.org/doi/10.1145/3534678.3539347
    - Code: https://github.com/liugangcode/GREA

    :param gamma: GREA-specific parameter. Default is 0.4.
    :type gamma: float
    """
    
    # GREA-specific parameter
    gamma: float = 0.4
    # Override parent defaults
    model_name: str = "GREAMolecularPredictor"
    model_class: Type[GREA] = field(default=GREA, init=False)
    
    def __post_init__(self):
        super().__post_init__()

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
            return {"prediction": np.array([]), "variance": np.array([]), "node_importance": np.array([])}