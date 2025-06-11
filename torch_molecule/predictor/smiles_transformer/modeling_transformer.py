import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, List, Callable, Type, Tuple
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import Transformer
from ..lstm.modeling_lstm import LSTMMolecularPredictor
from ...utils.search import (
    suggest_parameter,
    ParameterSpec,
    ParameterType,
)

# Dictionary mapping parameter names to their types and ranges
DEFAULT_TRANSFORMER_SEARCH_SPACES: Dict[str, ParameterSpec] = {
    # Integer-valued parameters
    "hidden_size": ParameterSpec(ParameterType.INTEGER, (64, 256)),
    "n_heads": ParameterSpec(ParameterType.INTEGER, (2, 8)),
    "num_layers": ParameterSpec(ParameterType.INTEGER, (2, 6)),
    "dim_feedforward": ParameterSpec(ParameterType.INTEGER, (128, 512)),
    # Float-valued parameters with log scale
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-5, 1e-2)),
    "weight_decay": ParameterSpec(ParameterType.LOG_FLOAT, (1e-8, 1e-3)),
    "dropout": ParameterSpec(ParameterType.FLOAT, (0.0, 0.5)),
    "scheduler_factor": ParameterSpec(ParameterType.FLOAT, (0.1, 0.5)),
}

@dataclass
class SMILESTransformerMolecularPredictor(LSTMMolecularPredictor):
    """This predictor implements a Transformer model for SMILES-based molecular property predictions.
    
    Notes
    -----
    This implementation uses a transformer encoder architecture to learn
    representations of molecular structures from SMILES strings.

    Parameters
    ----------
    num_task : int, default=1
        Number of prediction tasks.
    task_type : str, default="regression"
        Type of prediction task, either "regression" or "classification".
    input_dim : int, default=54
        Size of vocabulary for SMILES tokenization.
    hidden_size : int, default=128
        Dimension of embedding vectors.
    n_heads : int, default=4
        Number of attention heads in transformer layers.
    num_layers : int, default=3
        Number of transformer encoder layers.
    dim_feedforward : int, default=256
        Dimension of the feedforward network in transformer layers.
    max_input_len : int, default=200
        Maximum length of input sequences. Shorter sequences will be padded.
    dropout : float, default=0.1
        Dropout rate for transformer layers.
    batch_size : int, default=64
        Number of samples per batch for training.
    epochs : int, default=200
        Maximum number of training epochs.
    loss_criterion : callable, optional
        Loss function for training. Defaults to MSELoss for regression.
    evaluate_criterion : str or callable, optional
        Metric for model evaluation.
    evaluate_higher_better : bool, optional
        Whether higher values of the evaluation metric are better.
    learning_rate : float, default=0.0001
        Learning rate for optimizer.
    weight_decay : float, default=0.0
        L2 regularization strength.
    patience : int, default=20
        Number of epochs to wait for improvement before early stopping.
    use_lr_scheduler : bool, default=True
        Whether to use learning rate scheduler.
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate when plateau is reached.
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced.
    verbose : bool, default=False
        Whether to print progress information during training.
    """
    # Model parameters
    num_task: int = 1
    task_type: str = "regression"
    input_dim: int = 54  # vocabulary size
    hidden_size: int = 128
    n_heads: int = 4
    num_layers: int = 3
    dim_feedforward: Optional[int] = 256
    max_input_len: int = 200  # max token length
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 200
    loss_criterion: Optional[Callable] = None
    evaluate_criterion: Optional[Union[str, Callable]] = None
    evaluate_higher_better: Optional[bool] = None
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    patience: int = 20

    # Scheduler parameters
    use_lr_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Other parameters
    verbose: bool = False
    model_name: str = "SMILESTransformerMolecularPredictor"
    
    # Non-init fields
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    model_class: Type[Transformer] = field(default=Transformer, init=False)

    def __post_init__(self):
        """Initialize after dataclass initialization."""
        super().__post_init__()
        # Validate n_heads is compatible with hidden_size
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by n_heads ({self.n_heads})")
            
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
            "hidden_size",
            "n_heads",
            "num_layers",
            "dim_feedforward",
            "max_input_len",
            "dropout",
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
        """Get model parameters from checkpoint or current instance.
        
        Parameters
        ----------
        checkpoint : Optional[Dict], default=None
            Optional dictionary containing model checkpoint data
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters
        """
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
                
            hyperparameters = checkpoint["hyperparameters"]
            

            return {
                "num_task": hyperparameters.get("num_task", self.num_task),
                "input_dim": hyperparameters.get("input_dim", self.input_dim),
                "hidden_size": hyperparameters.get("hidden_size", self.hidden_size),
                "n_heads": hyperparameters.get("n_heads", self.n_heads),
                "num_layers": hyperparameters.get("num_layers", self.num_layers),
                "dim_feedforward": hyperparameters.get("dim_feedforward", self.dim_feedforward),
                "max_input_len": hyperparameters.get("max_input_len", self.max_input_len),
                "dropout": hyperparameters.get("dropout", self.dropout),
            }
        else:
            return {
                "num_task": self.num_task,
                "input_dim": self.input_dim,
                "hidden_size": self.hidden_size,
                "n_heads": self.n_heads,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "max_input_len": self.max_input_len,
                "dropout": self.dropout,
            }

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimization components including optimizer and learning rate scheduler.

        Returns
        -------
        Tuple[optim.Optimizer, Optional[Any]]
            A tuple containing:
            - The configured optimizer
            - The learning rate scheduler (if enabled, else None)
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min' if not self.evaluate_higher_better else 'max',
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
        
        return optimizer, scheduler


    def _get_default_search_space(self):
        """Get the default hyperparameter search space.
        """
        return DEFAULT_TRANSFORMER_SEARCH_SPACES