import warnings
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, List, Callable, Literal
import copy

import torch
from torch_geometric.loader import DataLoader

from .strategy import build_selection_dataset, build_augmentation_dataset
from ..grea.modeling_grea import GREAMolecularPredictor
from ..grea.model import GREA

from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

class SGIRMolecularPredictor(GREAMolecularPredictor):
    """This predictor implements SGIR for semi-supervised graph imbalanced regression.

    It trains the GREA model based on pseudo-labeling and data augmentation.

    References
    ----------
    - Semi-Supervised Graph Imbalanced Regression.
      https://dl.acm.org/doi/10.1145/3580305.3599497
    - Code: https://github.com/liugangcode/SGIR

    Parameters
    ----------
    num_anchor : int, default=10
        Number of anchor points used to split the label space during pseudo-labeling.
    warmup_epoch : int, default=20
        Number of epochs to train before starting pseudo-labeling and data augmentation.
    labeling_interval : int, default=5
        Interval (in epochs) between two pseudo-labeling steps. It controls the update frequency of pseudo-labeling.
    augmentation_interval : int, default=5
        Interval (in epochs) between two data augmentation steps. It controls the update frequency of data augmentation.
    top_quantile : float, default=0.1
        Quantile threshold for selecting high confidence predictions during pseudo-labeling.
    label_logscale : bool, default=False
        Whether to use log scale for the label space during pseudo-labeling and data augmentation.
    lw_aug : float, default=1
        Loss weight for the augmented data.
    gamma : float, default=0.4
        GREA-specific parameter that penalize the size of the rationales (ratio between the number of nodes in the rationales and the number of nodes in the original graph).
    num_task : int, default=1
        Number of prediction tasks. SGIR currently only supports regression tasks with 1 task.
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
        Device to run the model on. If None, will auto-detect GPU or use CPU.
    model_name : str, default="SGIRMolecularPredictor"
        Name identifier for the model.
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
        
        # SGIR-specific parameters
        self.num_anchor = num_anchor
        self.warmup_epoch = warmup_epoch
        self.labeling_interval = labeling_interval
        self.augmentation_interval = augmentation_interval
        self.top_quantile = top_quantile
        self.label_logscale = label_logscale
        self.lw_aug = lw_aug

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

        # Calculate total steps for global progress bar
        steps_per_epoch = len(train_loader)
        total_steps = self.epochs * steps_per_epoch
        
        # Initialize global progress bar
        global_pbar = None
        if self.verbose == "progress_bar":
            global_pbar = tqdm(
                total=total_steps,
                desc="SGIR Training Progress",
                unit="step",
                dynamic_ncols=True,
                leave=True
            )

        self.model.train()
        try:
            for epoch in range(self.epochs):
                # Training phase
                train_losses = self._train_epoch(train_loader, augmented_dataset, optimizer, epoch, global_pbar)
                
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
                    best_state_dict = copy.deepcopy(self.model.state_dict()) #Save the best epoch model not the last one
                    cnt_wait = 0
                    log_dict = {
                            "Epoch": f"{epoch+1}/{self.epochs}",
                            "Loss": f"{np.mean(train_losses):.4f}",
                            f"{self.evaluate_name}": f"{best_eval:.4f}",
                            "Status": "✓ Best"
                    }
                    if self.verbose == "progress_bar" and global_pbar:
                        global_pbar.set_postfix(log_dict)
                    elif self.verbose == "print_statement":
                        print(log_dict)
                else:
                    cnt_wait += 1
                    log_dict = {
                            "Epoch": f"{epoch+1}/{self.epochs}",
                            "Loss": f"{np.mean(train_losses):.4f}",
                            f"{self.evaluate_name}": f"{current_eval:.4f}",
                            "Wait": f"{cnt_wait}/{self.patience}"
                        }
                    if self.verbose == "progress_bar" and global_pbar:
                        global_pbar.set_postfix(log_dict)
                    elif self.verbose == "print_statement":
                        print(log_dict)
                    if cnt_wait > self.patience:
                        if self.verbose == "progress_bar":
                            if global_pbar:
                                global_pbar.set_postfix({
                                    "Status": "Early Stopped",
                                    "Epoch": f"{epoch+1}/{self.epochs}"
                                })
                        break
        finally:
            # Ensure progress bar is closed
            if global_pbar is not None:
                global_pbar.close()

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
    
    def _train_epoch(self, train_loader, augmented_dataset, optimizer, epoch, global_pbar=None):
        """Training logic for one epoch.

        Args:
            train_loader: DataLoader containing training data
            augmented_dataset: Augmented dataset for SGIR training
            optimizer: Optimizer instance for model parameter updates
            epoch: Current epoch number
            global_pbar: Global progress bar for tracking overall training progress

        Returns:
            list: List of loss values for each training step
        """
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

        for batch_idx, batch in enumerate(train_loader):
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

            # Update global progress bar
            if global_pbar is not None:
                global_pbar.update(1)
                global_pbar.set_postfix({
                    "Epoch": f"{epoch+1}/{self.epochs}",
                    "Batch": f"{batch_idx+1}/{len(train_loader)}",
                    "Total Loss": f"{loss.item():.4f}",
                    "(Pseudo)Labeled Loss": f"{Lx.item():.4f}",
                    "Aug Loss": f"{Laug.item():.4f}"
                })

        return losses