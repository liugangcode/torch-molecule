import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal
from dataclasses import dataclass

import torch
from torch_geometric.loader import DataLoader

from .strategy import build_selection_dataset, build_augmentation_dataset
from ..grea.modeling_grea import GREAMolecularPredictor
from ..grea.model import GREA

from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

@dataclass
class SGIRMolecularPredictor(GREAMolecularPredictor):
    """
    This predictor implements SGIR for semi-supervised graph imbalanced regression.

    It trains the GREA model based on pseudo-labeling and data augmentation.

    References
    ----------
    - Semi-Supervised Graph Imbalanced Regression.
      https://dl.acm.org/doi/10.1145/3580305.3599497
    - Code: https://github.com/liugangcode/SGIR

    :param num_anchor: Number of anchor points used to split the label space during pseudo-labeling
    :type num_anchor: int, default=10
    :param warmup_epoch: Number of epochs to train before starting pseudo-labeling and data augmentation
    :type warmup_epoch: int, default=20
    :param labeling_interval: Interval (in epochs) between pseudo-labeling steps
    :type labeling_interval: int, default=5
    :param augmentation_interval: Interval (in epochs) between data augmentation steps
    :type augmentation_interval: int, default=5
    :param top_quantile: Quantile threshold for selecting high confidence predictions during pseudo-labeling
    :type top_quantile: float, default=0.1
    :param label_logscale: Whether to use log scale for the label space during pseudo-labeling and data augmentation
    :type label_logscale: bool, default=False
    :param lw_aug: Weight for the data augmentation loss
    :type lw_aug: float, default=1
    """
    # SGIR-specific parameters
    num_anchor: int = 10
    warmup_epoch: int = 20
    labeling_interval: int = 5
    augmentation_interval: int = 5
    top_quantile: float = 0.1
    label_logscale: bool = False
    lw_aug: float = 1
    # Override parent defaults
    task_type: str = "regression"
    model_name: str = "SGIRMolecularPredictor"
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.task_type != "regression" or self.num_task != 1:
            raise ValueError("SGIR only supports regression tasks with 1 task")
        
    @staticmethod
    def _get_param_names():
        grea_params = [
            "num_anchor", "warmup_epoch", "labeling_interval",
            "augmentation_interval", "top_quantile", "label_logscale", "lw_aug"
        ]
        return grea_params + GREAMolecularPredictor._get_param_names()

    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["num_anchor"] = ParameterSpec(ParameterType.INTEGER, (10, 100))
        search_space["labeling_interval"] = ParameterSpec(ParameterType.INTEGER, (10, 20))
        search_space["augmentation_interval"] = ParameterSpec(ParameterType.INTEGER, (10, 20))
        search_space["top_quantile"] = ParameterSpec(ParameterType.LOG_FLOAT, (0.01, 0.5))
        search_space["lw_aug"] = ParameterSpec(ParameterType.FLOAT, (0.1, 1))
        return search_space

    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]],
        X_val: Optional[List[str]] = None,
        y_val: Optional[Union[List, np.ndarray]] = None,
        X_unlbl: Optional[List[str]] = None,
    ) -> "SGIRMolecularPredictor":
        """Fit the model to training data with optional validation set.
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be provided for validation")
        if X_unlbl is None:
            raise ValueError("X_unlbl (unlabeled SMILES strings) must be provided in SGIR")
        if len(X_unlbl) == 0:
            raise ValueError("X_unlbl (unlabeled SMILES strings) must not be empty")

        # Initialize model and optimization
        self._initialize_model(self.model_class)
        self.model.initialize_parameters()
        optimizer, scheduler = self._setup_optimizers()
        
        # Prepare datasets
        X_train, y_train = self._validate_inputs(X_train, y_train)
        train_dataset = self._convert_to_pytorch_data(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        X_unlbl, _ = self._validate_inputs(X_unlbl, None)
        unlbl_dataset = self._convert_to_pytorch_data(X_unlbl)

        if X_val is None:
            val_loader = train_loader
            warnings.warn(
                "No validation set provided. Using training set for validation.",
                UserWarning
            )
        else:
            X_val, y_val = self._validate_inputs(X_val, y_val)
            val_dataset = self._convert_to_pytorch_data(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

        # Training loop
        augmented_dataset = None
        self.fitting_loss = []
        self.fitting_epoch = 0
        best_state_dict = None
        best_eval = float('-inf') if self.evaluate_higher_better else float('inf')
        cnt_wait = 0

        self.model.train()
        for epoch in range(self.epochs):
            # Training phase
            train_losses = self._train_epoch(train_loader, augmented_dataset, optimizer, epoch)
            
            # Update datasets after warmup
            if epoch > self.warmup_epoch:
                if epoch % self.labeling_interval == 0:
                    train_loader = build_selection_dataset(
                        self.model, train_dataset, unlbl_dataset,
                        self.batch_size, self.num_anchor, self.top_quantile,
                        self.device, self.label_logscale
                    )

                if epoch % self.augmentation_interval == 0:
                    augmented_dataset = build_augmentation_dataset(
                        self.model, train_dataset, unlbl_dataset,
                        self.batch_size, self.num_anchor, self.device, 
                        self.label_logscale
                    )

            self.fitting_loss.append(np.mean(train_losses))

            # Validation and model selection
            current_eval = self._evaluation_epoch(val_loader)
            if scheduler:
                scheduler.step(current_eval)
            
            is_better = (
                current_eval > best_eval if self.evaluate_higher_better
                else current_eval < best_eval
            )
            
            if is_better:
                self.fitting_epoch = epoch
                best_eval = current_eval
                best_state_dict = self.model.state_dict()
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait > self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if self.verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Loss = {np.mean(train_losses):.4f}, "
                    f"{self.evaluate_name} = {current_eval:.4f}, "
                    f"Best {self.evaluate_name} = {best_eval:.4f}"
                )

        # Restore best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        else:
            warnings.warn(
                "No improvement achieved during training.",
                UserWarning
            )

        self.is_fitted_ = True
        return self
    
    def _train_epoch(self, train_loader, augmented_dataset, optimizer, epoch):
        losses = []

        if augmented_dataset is not None and self.lw_aug != 0:
            aug_reps = augmented_dataset['representations']
            aug_targets = augmented_dataset['labels']
            random_inds = torch.randperm(aug_reps.size(0))
            aug_reps = aug_reps[random_inds]
            aug_targets = aug_targets[random_inds]
            num_step = len(train_loader)
            aug_batch_size = aug_reps.size(0) // max(1, num_step)
            aug_inputs = list(torch.split(aug_reps, aug_batch_size))
            aug_outputs = list(torch.split(aug_targets, aug_batch_size))
        else:
            aug_inputs = None
            aug_outputs = None

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )

        for batch_idx, batch in enumerate(iterator):
            batch = batch.to(self.device)
            optimizer.zero_grad()

            # augmentation loss
            if aug_inputs is not None and aug_outputs is not None and aug_inputs[batch_idx].size(0) != 1:
                self.model._disable_batchnorm_tracking(self.model)
                pred_aug = self.model.predictor(aug_inputs[batch_idx])
                self.model._enable_batchnorm_tracking(self.model)
                targets_aug = aug_outputs[batch_idx]
                Laug = self.loss_criterion(pred_aug.view(targets_aug.size()).to(torch.float32), targets_aug).mean()
            else:
                Laug = torch.tensor(0.)      
            Lx = self.model.compute_loss(batch, self.loss_criterion)
            loss = Lx + Laug * self.lw_aug

            loss.backward()

            # Compute gradient norm if gradient clipping is enabled
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            optimizer.step()

            losses.append(loss.item())

            # Update progress bar if using tqdm
            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Total Loss": f"{loss.item():.4f}", "Lbls Loss": f"{Lx.item():.4f}", "Aug Loss": f"{Laug.item():.4f}",})

        return losses