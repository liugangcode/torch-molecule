import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal, Type
from dataclasses import dataclass, field

import torch
from torch_geometric.loader import DataLoader

from .model import DIR
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

@dataclass
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
    """
    
    # DIR-specific parameters
    causal_ratio: float = 0.8
    lw_invariant: float = 1e-4
    # Override parent defaults
    model_name: str = "DIRMolecularPredictor"
    model_class: Type[DIR] = field(default=DIR, init=False)
    
    def __post_init__(self):
        super().__post_init__()

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

    def _train_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        losses = []

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )

        alpha_prime = self.lw_invariant * (epoch ** 1.6)
        conf_opt = optimizer["conf"]
        model_optimizer = optimizer["model"]

        for batch in iterator:

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
            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Causal Loss": f"{causal_loss.item():.4f}", "Conf Loss": f"{conf_loss.item():.4f}", "Env Loss": f"{env_loss.item():.4f}", "Total Loss": f"{loss.item():.4f}"})

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
            for batch in tqdm(loader, disable=not self.verbose):
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


