import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal

import torch
from torch_geometric.loader import DataLoader

from .strategy import build_selection_dataset, build_augmentation_dataset
from ..grea.modeling_grea import GREAMolecularPredictor
from ..grea.architecture import GREA

from ...utils.search import (
    DEFAULT_GNN_SEARCH_SPACES,
    ParameterSpec,
    ParameterType,
)

class SGIRMolecularPredictor(GREAMolecularPredictor):
    """This predictor implements a GNN model based on pseudo-labeling and data augmentation.
    Paper: Semi-Supervised Graph Imbalanced Regression (https://dl.acm.org/doi/10.1145/3580305.3599497)
    Reference Code: https://github.com/liugangcode/SGIR
    """
    def __init__(
        self,
        # SGIR-specific parameters
        num_anchor: int = 10,
        warmup_epoch: int = 20,
        labeling_interval: int = 5,
        augmentation_interval: int = 5,
        top_quantile: float = 0.1,
        label_logscale: bool = False,
        lw_aug: float = 1,
        # GREA parameters
        gamma: float = 0.4,
        # model parameters
        num_task: int = 1,
        task_type: str = "regression",
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
        model_name: str = "SGIRMolecularPredictor",
    ):
        super().__init__(
            gamma=gamma,
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
        assert task_type == 'regression'
        # SGIR-specific parameter
        self.lw_aug = lw_aug
        self.num_anchor = num_anchor
        self.warmup_epoch = warmup_epoch
        self.labeling_interval = labeling_interval
        self.augmentation_interval = augmentation_interval
        self.top_quantile = top_quantile
        self.label_logscale = label_logscale
        self.model_class = GREA

    @staticmethod
    def _get_param_names():
        grea_params = [
            "num_anchor", "warmup_epoch", "labeling_interval",
            "augmentation_interval", "top_quantile", "label_logscale", "lw_aug"
        ]
        return grea_params + GREAMolecularPredictor._get_param_names()

    def _get_default_search_space(self):
        search_space = super()._get_default_search_space()
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

        # Initialize model and optimization
        self._initialize_model(self.model_class, self.device)
        self.model.initialize_parameters()
        self.model = self.model.to(self.device)
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
        unlbl_loader = DataLoader(
            unlbl_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

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
            train_losses = self._train_epoch(train_loader, augmented_dataset, optimizer)
            
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
    
    def _train_epoch(self, train_loader, augmented_dataset, optimizer):
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
                Laug = self.loss_criterion(pred_aug.view(targets_aug.size()).to(torch.float32), targets_aug)
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
                iterator.set_postfix({"loss": f"{loss.item():.4f}", "loss X": f"{Lx.item():.4f}", "loss aug": f"{Laug.item():.4f}",})

        return losses