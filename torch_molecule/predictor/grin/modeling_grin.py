from typing import Optional, Union, Dict, Any, List, Callable, Literal

import torch
import numpy as np
import warnings
import copy
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from .model import GRIN
from .utils import SmilesRepeat
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils import graph_from_smiles
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

class GRINMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements GRIN for Max Spanning Tree algorithm aligned GNN.

    The full name of GRIN is Graph Invariant Representation Learning.

    References
    ----------
    - Learning Repetition-Invariant Representations for Polymer Informatics. NeurIPS 2025
      https://arxiv.org/pdf/2505.10726

    Parameters
    ----------
    repetition_augmentation : bool, default=False
        Whether to enable polymer augmentation for training.
    l1_penalty : float, default=1e-3
        Weight for the L1 penalty.
    epochs_to_penalize : int, default=100
        Number of epochs to train before starting L1 penalty.
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
    device : torch.device or str, optional
        Device to use for computation.
    model_name : str, default="GRINMolecularPredictor"
        Name of the model.
    """
    def __init__(
        self,
        # GRIN-specific parameters
        repetition_augmentation: bool = False,
        l1_penalty: float = 1e-3,
        epochs_to_penalize: int = 100,
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
        model_name: str = "GRINMolecularPredictor"
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
        
        # GRIN-specific parameters
        self.repetition_augmentation = repetition_augmentation
        self.l1_penalty = l1_penalty
        self.epochs_to_penalize = epochs_to_penalize
        self.model_class = GRIN

        # Check CombineMols dependency if polymer augmentation is enabled
        if self.repetition_augmentation:
            try:
                from CombineMols.CombineMols import CombineMols
            except ImportError:
                raise ImportError(
                    "CombineMols is required for repetition augmentation for polymer. "
                    "Please install it using: pip install CombineMols"
                )

    
    @staticmethod
    def _get_param_names() -> List[str]:
        return GNNMolecularPredictor._get_param_names() + [
            "l1_penalty",
            "epochs_to_penalize"
        ]
    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["l1_penalty"] = ParameterSpec(ParameterType.LOG_FLOAT, (1e-6, 1))
        search_space["epochs_to_penalize"] = ParameterSpec(ParameterType.INTEGER, (0, 100))
        return search_space

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        base_params = super()._get_model_params(checkpoint)
        return base_params

    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]],
        X_val: Optional[List[str]] = None,
        y_val: Optional[Union[List, np.ndarray]] = None,
        X_unlbl: Optional[List[str]] = None,
    ) -> "GRINMolecularPredictor":
        """Fit the model to the training data with optional validation set.

        Parameters
        ----------
        X_train : List[str]
            Training set input molecular structures as SMILES strings
        y_train : Union[List, np.ndarray]
            Training set target values for property prediction
        X_val : List[str], optional
            Validation set input molecular structures as SMILES strings.
            If None, training data will be used for validation
        y_val : Union[List, np.ndarray], optional
            Validation set target values. Required if X_val is provided
        X_unlbl : List[str], optional
            Unlabeled set input molecular structures as SMILES strings.
            
        Returns
        -------
        self : GRINMolecularPredictor
            Fitted estimator
        """
        if (X_val is None) != (y_val is None):
            raise ValueError(
                "Both X_val and y_val must be provided for validation. "
                f"Got X_val={X_val is not None}, y_val={y_val is not None}"
            )

        self._initialize_model(self.model_class)
        self.model.initialize_parameters()
        optimizer, scheduler = self._setup_optimizers()
        
        # Prepare datasets and loaders
        if self.repetition_augmentation:
            X_train_aug, y_train_aug = SmilesRepeat(2).repeat(X_train, y_train)
            X_train = X_train + X_train_aug
            if y_train_aug is not None:
                if isinstance(y_train, np.ndarray):
                    y_train = np.concatenate([y_train, np.array(y_train_aug)], axis=0)
                else:
                    y_train = list(y_train) + list(y_train_aug)

        X_train, y_train = self._validate_inputs(X_train, y_train)
        train_dataset = self._convert_to_pytorch_data(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        if X_val is None or y_val is None:
            val_loader = train_loader
            warnings.warn(
                "No validation set provided. Using training set for validation. "
                "This may lead to overfitting.",
                UserWarning
            )
        else:
            if self.repetition_augmentation:
                X_val_aug, y_val_aug = SmilesRepeat(2).repeat(X_val, y_val)
                X_val = X_val + X_val_aug
                if y_val_aug is not None:
                    if isinstance(y_val, np.ndarray):
                        y_val = np.concatenate([y_val, np.array(y_val_aug)], axis=0)
                    else:
                        y_val = list(y_val) + list(y_val_aug)

            X_val, y_val = self._validate_inputs(X_val, y_val)
            val_dataset = self._convert_to_pytorch_data(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

        # Initialize training state
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
                desc="Training Progress",
                unit="step",
                dynamic_ncols=True
            )

        for epoch in range(self.epochs):
            # Training phase
            train_losses = self._train_epoch(train_loader, optimizer, epoch, global_pbar)
            self.fitting_loss.append(float(np.mean(train_losses)))

            # Validation phase
            current_eval = self._evaluation_epoch(val_loader)
            
            if scheduler:
                scheduler.step(current_eval)
            
            # Model selection (check if current evaluation is better)
            is_better = (
                current_eval > best_eval if self.evaluate_higher_better
                else current_eval < best_eval
            )
            
            if is_better:
                self.fitting_epoch = epoch
                best_eval = current_eval
                best_state_dict = copy.deepcopy(self.model.state_dict()) # Save the best epoch model not the last one
                cnt_wait = 0
                log_dict = {
                        "Epoch": f"{epoch+1}/{self.epochs}",
                        "Loss": f"{float(np.mean(train_losses)):.4f}",
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
                        "Loss": f"{float(np.mean(train_losses)):.4f}",
                        f"{self.evaluate_name}": f"{current_eval:.4f}",
                        "Wait": f"{cnt_wait}/{self.patience}"
                    }
                if self.verbose == "progress_bar" and global_pbar:
                    global_pbar.set_postfix(log_dict)
                elif self.verbose == "print_statement":    
                    print(log_dict)
                if cnt_wait > self.patience:
                    log_dict = {
                            "Status": "Early Stopped",
                            "Epoch": f"{epoch+1}/{self.epochs}"
                        }
                    if self.verbose == "progress_bar" and global_pbar:
                        global_pbar.set_postfix(log_dict)
                        global_pbar.close()
                    elif self.verbose == "print_stament":
                        print(log_dict)
                    break

        # Close global progress bar
        if global_pbar is not None:
            global_pbar.close()

        # Restore best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        else:
            warnings.warn(
                "No improvement was achieved during training. "
                "The model may not be fitted properly.",
                UserWarning
            )

        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, epoch, global_pbar=None):
        self.model.train()
        losses = []

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            if epoch >= self.epochs_to_penalize:
                l1_penalty = min(epoch - self.epochs_to_penalize, 1) * self.l1_penalty
            else:
                l1_penalty = 0
            loss = self.model.compute_loss(batch, self.loss_criterion, l1_penalty)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()

            if global_pbar is not None:
                global_pbar.update(1)
                global_pbar.set_postfix({
                    "Epoch": f"{epoch+1}/{self.epochs}",
                    "Batch": f"{batch_idx+1}/{len(train_loader)}",
                    "Loss": f"{loss.item():.4f}"
                })
            losses.append(loss.item())

        return losses

    def predict(self, X: List[str], test_time_augmentation: bool = False) -> Dict[str, np.ndarray]:
        """Make predictions using the fitted model.

        Parameters
        ----------
        X : List[str]
            List of SMILES strings to make predictions for
        test_time_augmentation : bool, default=False
            Whether to enable polymer augmentation for making predictions.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
                - 'prediction': Model predictions (shape: [n_samples, n_tasks])

        """
        self._check_is_fitted()
        if test_time_augmentation:
            X_aug, _ = SmilesRepeat(2).repeat(X)
            X = X_aug

        # Convert to PyTorch Geometric format and create loader
        X, _ = self._validate_inputs(X)
        dataset = self._convert_to_pytorch_data(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.model is None:
            raise RuntimeError("Model not initialized")
        # Make predictions
        self.model = self.model.to(self.device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            if self.verbose == "progress_bar":
                iterator = tqdm(loader, desc="Predicting")
            elif self.verbose == "print_statement":
                print("Predicting...")
                iterator = loader
            else:
                iterator = loader
            for batch in iterator:
                batch = batch.to(self.device)
                out = self.model(batch)
                predictions.append(out["prediction"].cpu().numpy())
        return {
            "prediction": np.concatenate(predictions, axis=0),
        }