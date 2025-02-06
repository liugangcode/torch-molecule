import os
import json
import datetime
import tempfile
import warnings
import torch
import numpy as np
from rdkit import Chem
from ..utils import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from ..utils.checkpoint import LocalCheckpointManager, HuggingFaceCheckpointManager
from ..utils.checker import MolecularInputChecker
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, ClassVar, Union, List, Dict, Any, Tuple, Callable, Type

@dataclass
class BaseMolecularPredictor(ABC):
    """Base class for molecular discovery estimators.
    
    This class provides a structure for implementing custom molecular discovery
    models. It includes basic functionality for parameter management, fitting,
    prediction, and validation.
    """

    num_task: int
    task_type: str
    model_name: str = field(default="BaseMolecularPredictor")
    device: Optional[torch.device] = field(default=None)
    model: Optional[torch.nn.Module] = field(default=None, init=False)

    DEFAULT_METRICS: ClassVar[Dict] = {
        "classification": {
            "default": ("roc_auc", roc_auc_score, True),
        },
        "regression": {"default": ("mae", mean_absolute_error, False)},
    }

    is_fitted_: bool = field(default=False, init=False)
    model_class: Optional[Type[torch.nn.Module]] = field(default=None, init=False)

    def __post_init__(self):
        if self.task_type not in ["classification", "regression"]:
            raise ValueError(
                f"task_type must be one of ['classification', 'regression'], got {self.task_type}"
            )

        if not isinstance(self.num_task, (int, np.integer)) or self.num_task <= 0:
            raise ValueError(f"num_task must be a positive integer, got {self.num_task}")

        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params) -> "BaseMolecularPredictor":
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for predictor {self}")
            setattr(self, key, value)
        return self

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, X_unlbl=None) -> "BaseMolecularPredictor":
        pass

    @abstractmethod
    def autofit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        search_parameters: Optional[dict] = None,
        n_trials: int = 10,
    ) -> "BaseMolecularPredictor":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        raise NotImplementedError("Child class must implement optimizer and scheduler setup.")

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

    def _train_epoch(self, train_loader, optimizer):
        raise NotImplementedError("Child class must implement training logic.")

    def _evaluation_epoch(self, evaluate_loader):
        raise NotImplementedError("Child class must implement evaluation logic.")

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise AttributeError(
                "This instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    def __repr__(self, N_CHAR_MAX=700):
        attributes = {
            name: value
            for name, value in sorted(self.__dict__.items())
            if not name.startswith("_") and not callable(value)
        }

        def format_value(v):
            if isinstance(v, (float, np.float32, np.float64)):
                return f"{v:.3g}"
            elif isinstance(v, (list, tuple, np.ndarray)) and len(v) > 6:
                return f"{str(v[:3])[:-1]}, ..., {str(v[-3:])[1:]}"
            elif isinstance(v, str) and len(v) > 50:
                return f"'{v[:25]}...{v[-22:]}'"
            elif isinstance(v, dict) and len(v) > 6:
                items = list(v.items())
                return f"{dict(items[:3])}...{dict(items[-3:])}"
            elif isinstance(v, torch.nn.Module):
                return f"{v.__class__.__name__}(...)"
            elif v is None:
                return "None"
            else:
                return repr(v)

        class_name = self.__class__.__name__
        attributes_str = []

        important_attrs = ["num_task", "task_type", "model_name", "is_fitted_"]
        for attr in important_attrs:
            if attr in attributes:
                value = attributes.pop(attr)
                attributes_str.append(f"{attr}={format_value(value)}")

        sorted_attrs = sorted(attributes.items())
        attributes_str.extend(f"{name}={format_value(value)}" for name, value in sorted_attrs)

        content = ",\n    ".join(attributes_str)
        repr_ = f"{class_name}(\n    {content}\n)"

        if len(repr_) > N_CHAR_MAX:
            lines = repr_.split("\n")
            result = [lines[0]]
            current_length = len(lines[0])
            for line in lines[1:-1]:
                if current_length + len(line) + 5 > N_CHAR_MAX:
                    result.append("    ...")
                    break
                result.append(line)
                current_length += len(line) + 1
            result.append(lines[-1])
            repr_ = "\n".join(result)

        return repr_

    @abstractmethod
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        raise NotImplementedError("Child class must implement _get_model_params.")

    def _initialize_model(
        self, model_class: Type[torch.nn.Module], device, checkpoint: Optional[Dict] = None
    ) -> None:
        """Initialize the model with parameters or a checkpoint."""
        try:
            model_params = self._get_model_params(checkpoint)
            self.model = model_class(**model_params)
            try:
                self.model = self.model.to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to move model to device {device}: {str(e)}")

            if checkpoint is not None:
                try:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    raise ValueError(f"Failed to load model state dictionary: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _load_default_criterion(self):
        if self.task_type == "regression":
            return torch.nn.L1Loss()
        elif self.task_type == "classification":
            return torch.nn.BCEWithLogitsLoss()
        else:
            warnings.warn(
                "Unknown task type. Using L1 Loss as default. "
                "Please specify 'regression' or 'classification' for better results."
            )
            return torch.nn.L1Loss()

    def _validate_inputs(
        self, X: List[str], y: Optional[Union[List, np.ndarray]] = None, return_rdkit_mol: bool = True
    ) -> Tuple[Union[List[str], List["Chem.Mol"]], Optional[np.ndarray]]:
        return MolecularInputChecker.validate_inputs(X, y, self.num_task, 0, return_rdkit_mol)

    # def _validate_smiles(self, smiles: str, idx: int) -> Tuple[bool, Optional[str], Optional[Chem.Mol]]:
    #     return MolecularInputChecker.validate_smiles(smiles, idx)

    @staticmethod
    def _get_param_names() -> List[str]:
        return ["num_task", "task_type", "is_fitted_"]

    def save_to_local(self, path: str) -> None:
        """Save to local disk using LocalCheckpointManager."""
        LocalCheckpointManager.save_model_to_local(self, path)

    def load_from_local(self, path: str) -> None:
        """Load from local disk using LocalCheckpointManager."""
        LocalCheckpointManager.load_model_from_local(self, path)

    def save_to_hf(
        self,
        repo_id: str,
        task_id: str = "default",
        metadata_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        commit_message: str = "Update model",
        token: Optional[str] = None,
        private: bool = False,
    ) -> None:
        HuggingFaceCheckpointManager.push_to_huggingface(
            model_instance=self,
            repo_id=repo_id,
            task_id=task_id,
            metadata_dict=metadata_dict,
            metrics=metrics,
            commit_message=commit_message,
            token=token,
            private=private,
        )

    def load_from_hf(self, repo_id: str, path: str) -> None:
        """Download and load model from Hugging Face Hub."""
        HuggingFaceCheckpointManager.load_model_from_hf(self, repo_id, path)

    def save(self, path: Optional[str] = None, repo_id: Optional[str] = None, **kwargs) -> None:
        """
        Automatic save. If `repo_id` is given, push to Hugging Face Hub.
        Otherwise, save to local path.
        """
        if repo_id is not None:
            self.save_to_hf(repo_id=repo_id, **kwargs)
        else:
            if path is None:
                raise ValueError("path must be provided if repo_id is not given.")
            self.save_to_local(path)

    def load(self, path: str, repo_id: Optional[str] = None) -> None:
        """
        Automatic load. If local `path` exists, load from there. 
        Otherwise, try loading from Hugging Face Hub if `repo_id` is given.
        """
        if os.path.exists(path):
            self.load_from_local(path)
        else:
            if repo_id is None:
                raise FileNotFoundError(
                    f"No local file found at '{path}' and no repo_id provided."
                )
            self.load_from_hf(repo_id, path)
