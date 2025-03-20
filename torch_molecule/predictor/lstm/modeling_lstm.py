import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal, Type
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import LSTM
from .token import create_tensor_dataset
from ...base import BaseMolecularPredictor
from ...utils.search import (
    suggest_parameter,
    ParameterSpec,
    ParameterType,
    parse_list_params,
)

# Dictionary mapping parameter names to their types and ranges
DEFAULT_LSTM_SEARCH_SPACES: Dict[str, ParameterSpec] = {
    # Integer-valued parameters
    "output_dim": ParameterSpec(ParameterType.INTEGER, (8, 32)),
    "LSTMunits": ParameterSpec(ParameterType.INTEGER, (30, 120)),
    # Float-valued parameters with log scale
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-4, 1e-2)),
    "weight_decay": ParameterSpec(ParameterType.LOG_FLOAT, (1e-8, 1e-3)),
}

@dataclass
class LSTMMolecularPredictor(BaseMolecularPredictor):
    """This predictor implements a LSTM model for molecular property prediction tasks.
    Paper: Predicting Polymers' Glass Transition Temperature by a Chemical Language Processing Model (https://www.semanticscholar.org/reader/f43ed533b2520567be2d8c24f6396f4e63e96430)
    Reference Code: https://github.com/figotj/RNN-Tg
    """
    # Model parameters
    num_task: int = 1
    task_type: str = "regression"
    input_dim: int = 53 # vocabulary size
    output_dim: int = 15
    LSTMunits: int = 60
    max_input_len: int = 200 # max token length
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 500
    loss_criterion: Optional[Callable] = None
    evaluate_criterion: Optional[Union[str, Callable]] = None
    evaluate_higher_better: Optional[bool] = None
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    patience: int = 50

    # Scheduler parameters
    use_lr_scheduler: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Other parameters
    verbose: bool = False
    model_name: str = "LSTMMolecularPredictor"
    
    # Non-init fields
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    model_class: Type[LSTM] = field(default=LSTM, init=False)

    def __post_init__(self):
        # Setup loss criterion and evaluation
        if self.loss_criterion is None:
            self.loss_criterion = nn.MSELoss()
        self._setup_evaluation(self.evaluate_criterion, self.evaluate_higher_better)

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        List[str]
            List of parameter names that can be used for model configuration.
        """
        return [
            # Model Hyperparameters
            "num_task",
            "task_type",
            "input_dim",
            "output_dim",
            "LSTMunits",
            "max_input_len",
            # Training Parameters
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "patience",
            "loss_criterion",
            # Evaluation Parameters
            "evaluate_name",
            "evaluate_criterion",
            "evaluate_higher_better",
            # Scheduler Parameters
            "use_lr_scheduler",
            "scheduler_factor",
            "scheduler_patience",
            # Other Parameters
            "fitting_epoch",
            "fitting_loss",
            "device",
            "verbose"
        ]
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
                
            hyperparameters = checkpoint["hyperparameters"]
            
            return {
                "num_task": hyperparameters.get("num_task", self.num_task),
                "input_dim": hyperparameters.get("input_dim", self.input_dim),
                "output_dim": hyperparameters.get("output_dim", self.output_dim),
                "LSTMunits": hyperparameters.get("LSTMunits", self.LSTMunits),
                "max_input_len": hyperparameters.get("LSTMunits", self.max_input_len),
            }
        else:
            return {
                "num_task": self.num_task,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "LSTMunits": self.LSTMunits,
                "max_input_len": self.max_input_len,
            }

    def _convert_to_pytorch_data(self, X, y=None):
        """Convert numpy arrays to PyTorch data format.
        """
        if self.verbose:
            iterator = tqdm(enumerate(X), desc="Converting lists of data to tensordataset", total=len(X))
        else:
            iterator = enumerate(X)
        tokenized_X = create_tensor_dataset(X, self.max_input_len)
        if y is not None and y.size > 0:
            if len(y) != len(X):
                raise ValueError(f"The number of smiles {len(X)} is incompatible with the number of labels {len(y)}!")
            return TensorDataset(torch.tensor(tokenized_X, dtype=torch.long),
                                    torch.tensor(y, dtype=torch.float32))
        return TensorDataset(torch.tensor(tokenized_X, dtype=torch.long),
                                    torch.zeros(len(tokenized_X), dtype=torch.float32))

    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]],
        X_val: Optional[List[str]] = None,
        y_val: Optional[Union[List, np.ndarray]] = None,
    ) -> "LSTMMolecularPredictor":
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
        if not isinstance(X_train, list) or not all(isinstance(item, str) for item in X_train):
            raise TypeError(f"Expected train data to be a list of strings, but got {type(X_train)} with elements {X_train}")

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
            if not isinstance(X_val, list) or not all(isinstance(item, str) for item in X_val):
                raise TypeError(f"Expected valid data to be a list of strings, but got {type(X_val)} with elements {X_val}")

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

        for epoch in range(self.epochs):
            # Training phase
            train_losses = self._train_epoch(train_loader, optimizer)
            self.fitting_loss.append(np.mean(train_losses))

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
                best_state_dict = self.model.state_dict()
                cnt_wait = 0
            else:
                cnt_wait += 1
                if cnt_wait > self.patience:
                    if self.verbose:
                        print(f"Early stopping triggered after {epoch} epochs")
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
                "No improvement was achieved during training. "
                "The model may not be fitted properly.",
                UserWarning
            )

        self.is_fitted_ = True
        return self

    def predict(self, X: List[str]) -> Dict[str, np.ndarray]:
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

        """
        self._check_is_fitted()

        # Convert to token format and create loader
        if not isinstance(X, list) or not all(isinstance(item, str) for item in X):
            raise TypeError(f"Expected X to be a list of strings, but got {type(X)} with elements {X}")
        dataset = self._convert_to_pytorch_data(X) 
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.model is None:
            raise RuntimeError("Model not initialized")
        # Make predictions
        self.model = self.model.to(self.device)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(loader, disable=not self.verbose):
                batched_input, batched_label = batch
                batched_input = batched_input.to(self.device)
 
                out = self.model(batched_input)
                predictions.append(out["prediction"].cpu().numpy())
        return {
            "prediction": np.concatenate(predictions, axis=0),
        }

    def _evaluation_epoch(
        self,
        loader: DataLoader,
    ) -> float:
        """Evaluate the model on given data.
        
        Parameters
        ----------
        loader : DataLoader
            DataLoader containing evaluation data
        train_losses : List[float]
            Training losses from current epoch
            
        Returns
        -------
        float
            Evaluation metric value (adjusted for higher/lower better)
        """
        self.model.eval()
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for batch in loader:
                batched_input, batched_label = batch
                batched_input = batched_input.to(self.device)

                out = self.model(batched_input)
                y_pred_list.append(out["prediction"].detach().cpu().numpy())  # ensuring NumPy format
                y_true_list.append(batched_label.cpu().numpy())
        
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_pred = y_pred.reshape(-1, 1)  
        y_true = np.concatenate(y_true_list, axis=0)
        
        # Compute metric
        metric_value = float(self.evaluate_criterion(y_true, y_pred))
        
        # Adjust metric value based on higher/lower better
        return metric_value

    def _train_epoch(self, train_loader, optimizer):
        """Training logic for one epoch.

        Args:
            train_loader: DataLoader containing training data
            optimizer: Optimizer instance for model parameter updates

        Returns:
            list: List of loss values for each training step
        """
        self.model.train()
        losses = []

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )

        for batch in iterator:
            batched_input, batched_label = batch
            batched_input = batched_input.to(self.device)
            batched_label = batched_label.to(self.device).view(-1, 1)
            optimizer.zero_grad()

            # Forward pass and loss computation
            loss = self.model.compute_loss(batched_input, batched_label, self.loss_criterion)

            # Backward pass
            loss.backward()

            optimizer.step()

            losses.append(loss.item())

            # Update progress bar if using tqdm
            if self.verbose:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        return losses

