import os
import json
import datetime
import tempfile
import warnings
import torch
import numpy as np
from rdkit import Chem
from ..utils.checker import MolecularInputChecker
from ..utils.checkpoint import LocalCheckpointManager, HuggingFaceCheckpointManager
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, ClassVar, Union, List, Dict, Any, Tuple, Callable, Type, Literal

@dataclass
class BaseMolecularEncoder(ABC):
    """Base class for molecular representation learning.

    This class provides a skeleton for implementing custom molecular representation
    learning models. It includes basic functionality for parameter management, 
    fitting, and encoding molecular representations.

    Attributes
    ----------
    model_name : str
        Name of the model
    device : Optional[torch.device]
        Device to run the model on
    model : Optional[torch.nn.Module]
        Model instance
    """
    
    model_name: str = field(default="BaseMolecularEncoder")
    device: Optional[torch.device] = field(default=None)
    model: Optional[torch.nn.Module] = field(default=None, init=False)
    
    # Instance variables with default values
    is_fitted_: bool = field(default=False, init=False)
    model_class: Optional[Type[torch.nn.Module]] = field(default=None, init=False)
    
    def __post_init__(self):
        """Set device if not provided."""
        # Set device if not provided
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params) -> "BaseMolecularEncoder":
        """Set the parameters of this representer.

        Parameters
        ----------
        **params : dict
            Representer parameters.

        Returns
        -------
        self : object
            Representer instance.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for representer {self}")
            setattr(self, key, value)
        return self

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        raise NotImplementedError("optimizers and schedulers must be implemented by child class")

    @abstractmethod
    def fit(self, X: List[str], y: Optional[np.ndarray] = None) -> "BaseMolecularEncoder":
        pass

    @abstractmethod
    def encode(self, X: List[str], return_type: Literal["np", "pt"] = "pt") -> Union[np.ndarray, torch.Tensor]:
        pass

    def _train_epoch(self, train_loader, optimizer):
        raise NotImplementedError("Training epoch logic must be implemented by child class")

    def _evaluation_epoch(self, evaluate_loader):
        raise NotImplementedError("Evaluation epoch logic must be implemented by child class")

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise AttributeError(
                "This MolecularBaseRepresenter instance is not fitted yet. "
                "Call 'fit' before using this representer."
            )

    def __repr__(self, N_CHAR_MAX=700):
        """Get string representation of the representer.

        Parameters
        ----------
        N_CHAR_MAX : int, default=700
            Maximum number of non-blank characters to show

        Returns
        -------
        str
            String representation of the representer
        """
        # Get all relevant attributes (excluding private ones and callables)
        attributes = {
            name: value
            for name, value in sorted(self.__dict__.items())
            if not name.startswith("_") and not callable(value)
        }

        def format_value(v):
            """Format a value for representation."""
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

        # Build the representation string
        class_name = self.__class__.__name__
        attributes_str = []

        # First add important attributes that should appear first
        important_attrs = ["model_name", "is_fitted_"]
        for attr in important_attrs:
            if attr in attributes:
                value = attributes.pop(attr)
                attributes_str.append(f"{attr}={format_value(value)}")

        # Then add the rest
        sorted_attrs = sorted(attributes.items())
        attributes_str.extend(f"{name}={format_value(value)}" for name, value in sorted_attrs)

        # Join all parts
        content = ",\n    ".join(attributes_str)
        repr_ = f"{class_name}(\n    {content}\n)"

        # Handle length limit
        if len(repr_) > N_CHAR_MAX:
            lines = repr_.split("\n")

            # Keep header
            result = [lines[0]]
            current_length = len(lines[0])

            # Process middle lines
            for line in lines[1:-1]:
                if current_length + len(line) + 5 > N_CHAR_MAX:  # 5 for "...\n"
                    result.append("    ...")
                    break
                result.append(line)
                current_length += len(line) + 1  # +1 for newline

            # Add closing parenthesis
            result.append(lines[-1])
            repr_ = "\n".join(result)

        return repr_

    @abstractmethod
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        raise NotImplementedError(
            "Method _get_model_params must be implemented by child class. "
            "This method should return a dictionary of model parameters either "
            "from a checkpoint or current instance values."
        )

    def _initialize_model(
        self, model_class: Type[torch.nn.Module], device, checkpoint: Optional[Dict] = None
    ) -> None:
        """Initialize the model with given parameters or checkpoint.
        """
        try:
            model_params = self._get_model_params(checkpoint)

            # Initialize model with parameters
            self.model = model_class(**model_params)

            # Move model to device
            try:
                self.model = self.model.to(device)
            except Exception as e:
                raise RuntimeError(f"Failed to move model to device {device}: {str(e)}")

            # Load state dict if provided in checkpoint
            if checkpoint is not None:
                try:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as e:
                    raise ValueError(f"Failed to load model state dictionary: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    @staticmethod
    def _get_param_names() -> List[str]:
        return ["model_name", "is_fitted_"]
    
    def _validate_inputs(
        self, X: List[str], y: Optional[Union[List, np.ndarray]] = None, return_rdkit_mol: bool = True, num_task: int = 0, predefined_num_task: int = 0
    ) -> Tuple[Union[List[str], List["Mol"]], Optional[np.ndarray]]:
        return MolecularInputChecker.validate_inputs(X, y, num_task, predefined_num_task, return_rdkit_mol)

    def _inspect_task_types(self, y: Union[np.ndarray, torch.Tensor], return_type: Literal["pt", "np"] = "pt") -> Union[np.ndarray, torch.Tensor]:
        """Inspect the task types of the target values.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            Target values: 2D array for multiple tasks
        return_type : Literal["pt", "np"], default="pt"
            Return type of the result

        Returns
        -------
        result : Union[np.ndarray, torch.Tensor]
            Result of the task types inspection
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        # 0/1/nan -> binary classification (True), otherwise -> regression (False)
        result = torch.tensor([torch.all(torch.isnan(y[:, i]) | (y[:, i] == 0) | (y[:, i] == 1)) for i in range(y.shape[1])], dtype=torch.bool)
        if return_type == "np":
            return result.numpy()
        return result

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