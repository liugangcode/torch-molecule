import warnings
import torch
import numpy as np
from ..utils import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, ClassVar, Union, List, Dict, Any, Tuple, Callable, Type
from ..base.base import BaseModel

@dataclass
class BaseMolecularPredictor(BaseModel, ABC):
    """Base class for molecular discovery estimators."""
    
    model_name: str = field(default="BaseMolecularPredictor")
    num_task: int = field(default=0)
    task_type: str = field(default=None)
    DEFAULT_METRICS: ClassVar[Dict] = {
        "classification": {"default": ("roc_auc", roc_auc_score, True)},
        "regression": {"default": ("mae", mean_absolute_error, False)},
    }

    def __post_init__(self):
        super().__post_init__()
        if self.task_type not in ["classification", "regression"]:
            raise ValueError(f"Invalid task_type: {self.task_type}")
        if self.num_task <= 0:
            raise ValueError(f"num_task must be positive, got {self.num_task}")

    @staticmethod
    def _get_param_names(self) -> List[str]:
        return super()._get_param_names() + ["num_task", "task_type"]

    @abstractmethod
    def autofit(self, X_train, y_train, X_val=None, y_val=None, search_parameters: Optional[dict] = None, n_trials: int = 10) -> "BaseMolecularPredictor": 
        pass
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, search_parameters: Optional[dict] = None, n_trials: int = 10) -> "BaseMolecularPredictor": 
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _evaluation_epoch(self, evaluate_loader):
        pass
        
    def _setup_evaluation(
        self,
        evaluate_criterion: Optional[Union[str, Callable]],
        evaluate_higher_better: Optional[bool],
    ) -> None:
        if evaluate_criterion is None:
            default_metric = self.DEFAULT_METRICS[self.task_type]["default"]
            self.evaluate_name = default_metric[0]
            self.evaluate_criterion = default_metric[1]
            self.evaluate_higher_better = default_metric[2]
        else:
            if isinstance(evaluate_criterion, str):
                metric_map = {
                    "accuracy": (accuracy_score, True),
                    "roc_auc": (roc_auc_score, True),
                    "rmse": (root_mean_squared_error, False),
                    "mse": (mean_squared_error, False),
                    "mae": (mean_absolute_error, False),
                    "r2": (r2_score, True),
                }
                if evaluate_criterion not in metric_map:
                    raise ValueError(
                        f"Unknown metric: {evaluate_criterion}. "
                        f"Available metrics: {list(metric_map.keys())}"
                    )
                self.evaluate_name = evaluate_criterion
                self.evaluate_criterion = metric_map[evaluate_criterion][0]
                self.evaluate_higher_better = (
                    metric_map[evaluate_criterion][1]
                    if evaluate_higher_better is None
                    else evaluate_higher_better
                )
            else:
                if evaluate_higher_better is None:
                    raise ValueError(
                        "evaluate_higher_better must be specified for a custom function."
                    )
                self.evaluate_name = "custom"
                self.evaluate_criterion = evaluate_criterion
                self.evaluate_higher_better = evaluate_higher_better
    
    def _load_default_criterion(self):
        if self.task_type == "regression":
            return torch.nn.L1Loss(reduction='none')
        elif self.task_type == "classification":
            return torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            warnings.warn(
                "Unknown task type. Using L1 Loss as default. "
                "Please specify 'regression' or 'classification' for better results."
            )
            return torch.nn.L1Loss(reduction='none')
    
    def _validate_inputs(
        self, X: List[str], y: Optional[Union[List, np.ndarray]] = None, num_task: int = 0, num_pretask: int = 0, return_rdkit_mol: bool = True
    ) -> Tuple[Union[List[str], List["Chem.Mol"]], Optional[np.ndarray]]:
        return super()._validate_inputs(X, y, self.num_task, 0, return_rdkit_mol)