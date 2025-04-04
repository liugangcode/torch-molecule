from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Type, Any, ClassVar, Union, Tuple, Callable, Literal
import torch
import os
import numpy as np
from ..utils.checkpoint import LocalCheckpointManager, HuggingFaceCheckpointManager
from ..utils.checker import MolecularInputChecker

@dataclass
class BaseModel(ABC):
    """Base class for molecular models with shared functionality.
    
    This abstract class provides common methods and utilities for molecular models,
    including model initialization, saving/loading, and parameter management.
    """
    
    device: Optional[torch.device] = field(default=None)
    model_name: str = field(default="BaseModel")
    model_class: Optional[Type[torch.nn.Module]] = field(default=None, init=False) # used for model initialization
    model: Optional[torch.nn.Module] = field(default=None, init=False) # initialized model
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self):
        """Initialize common device settings after instance creation.
        
        Sets the device to CUDA if available, otherwise CPU, when no device is specified.
        Converts string device specifications to torch.device objects.
        """
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)

    @abstractmethod
    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Set up optimizers for model training.
        
        Returns
        -------
        Tuple[torch.optim.Optimizer, Optional[Any]]
            Tuple containing the primary optimizer and an optional secondary optimizer or scheduler.
        """
        pass
    
    @abstractmethod
    def _train_epoch(self, train_loader, optimizer):
        """Train the model for one epoch.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader containing training batches
        optimizer : torch.optim.Optimizer
            Optimizer to use for parameter updates
            
        Returns
        -------
        dict
            Training metrics for the epoch
        """
        pass

    @abstractmethod
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        """Get model parameters used for model initialization.
        
        Parameters
        ----------
        checkpoint : Optional[Dict], default=None
            Optional dictionary containing model checkpoint data
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of parameters to initialize the model
        """
        pass

    @staticmethod
    def _get_param_names(self) -> List[str]:
        """Get parameter names in the modeling class.
        
        Returns
        -------
        List[str]
            List of parameter names that can be configured
        """
        # return ["model_name", "model_class", "is_fitted_"]
        return ["model_name", "is_fitted_"]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
                 
        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter names mapped to their values
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params) -> "BaseModel":
        """Set parameters for this estimator.
        
        Parameters
        ----------
        **params
            Parameter names mapped to their values
            
        Returns
        -------
        BaseModel
            Self instance for method chaining
            
        Raises
        ------
        ValueError
            If an invalid parameter is provided
        """
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for model {self}")
            setattr(self, key, value)
        return self

    def _initialize_model(
        self,
        model_class: Type[torch.nn.Module],
        checkpoint: Optional[Dict] = None
    ) -> torch.nn.Module:
        """Initialize the model with parameters or a checkpoint.
        
        Parameters
        ----------
        model_class : Type[torch.nn.Module]
            PyTorch module class to instantiate
        checkpoint : Optional[Dict], default=None
            Optional dictionary containing model checkpoint data
            
        Returns
        -------
        torch.nn.Module
            Initialized PyTorch model
        """
        model_params = self._get_model_params(checkpoint)
        self.model = model_class(**model_params)
        self.model = self.model.to(self.device)
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        return self.model
    
    def _validate_inputs(
        self, X: List[str], y: Optional[Union[List, np.ndarray]] = None, num_task: int = 0, num_pretask: int = 0, return_rdkit_mol: bool = True
    ) -> Tuple[Union[List[str], List["Chem.Mol"]], Optional[np.ndarray]]:
        """Validate molecular inputs and targets.
        
        Parameters
        ----------
        X : List[str]
            List of SMILES strings representing molecules
        y : Optional[Union[List, np.ndarray]], default=None
            Optional target values for supervised learning
        num_task : int, default=0
            Number of prediction tasks
        num_pretask : int, default=0
            Number of pre-training tasks
        return_rdkit_mol : bool, default=True
            Whether to return RDKit Mol objects instead of SMILES
            
        Returns
        -------
        Tuple[Union[List[str], List["Chem.Mol"]], Optional[np.ndarray]]
            Tuple of validated inputs and targets
        """
        return MolecularInputChecker.validate_inputs(X, y, num_task, num_pretask, return_rdkit_mol)

    def save_to_local(self, path: str) -> None:
        """Save model to local disk.
        
        Parameters
        ----------
        path : str
            File path to save the model
            
        Raises
        ------
        ValueError
            If the model is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving to local disk.")
        LocalCheckpointManager.save_model_to_local(self, path)

    def load_from_local(self, path: str) -> None:
        """Load model from local disk.
        
        Parameters
        ----------
        path : str
            File path to load the model from
        """
        LocalCheckpointManager.load_model_from_local(self, path)

    def save_to_hf(
        self,
        repo_id: str,
        task_id: str = "default",
        metadata_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        commit_message: str = "Update model",
        hf_token: Optional[str] = None,
        private: bool = False,
        config_filename: Optional[str] = 'config.json',
    ) -> None:
        """Save model to Hugging Face Hub.
        
        Parameters
        ----------
        repo_id : str
            Hugging Face repository ID
        task_id : str, default="default"
            Task identifier for the model
        metadata_dict : Optional[Dict[str, Any]], default=None
            Optional metadata to store with the model
        metrics : Optional[Dict[str, float]], default=None
            Optional performance metrics to store with the model
        commit_message : str, default="Update model"
            Git commit message
        hf_token : Optional[str], default=None
            Hugging Face authentication token
        private : bool, default=False
            Whether the repository should be private
        config_filename : Optional[str], default='config.json'
            Name of the configuration file to save to the repository
        Raises
        ------
        ValueError
            If the model is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving to Hugging Face Hub.")
        HuggingFaceCheckpointManager.push_to_huggingface(
            model_instance=self,
            repo_id=repo_id,
            task_id=task_id,
            metadata_dict=metadata_dict,
            metrics=metrics,
            commit_message=commit_message,
            token=hf_token,
            private=private,
            config_filename=config_filename,
        )

    def load_from_hf(self, repo_id: str, local_cache: Optional[str] = None, config_filename: Optional[str] = 'config.json') -> None:
        """Load model from Hugging Face Hub.
        
        Parameters
        ----------
        repo_id : str
            Hugging Face repository ID
        local_cache : str, default=None
            Local path to save the model
        config_filename : str, default='config.json'
            Name of the configuration file to load from the repository
        """
        if local_cache is None:
            local_cache = 'model.pt'
        HuggingFaceCheckpointManager.load_model_from_hf(self, repo_id, local_cache, config_filename=config_filename)

    def save(self, path: Optional[str] = None, repo_id: Optional[str] = None, **kwargs) -> None:
        """Automatic save to either local disk or Hugging Face Hub.
        
        Parameters
        ----------
        path : Optional[str], default=None
            File path for local saving (required if repo_id is None)
        repo_id : Optional[str], default=None
            Hugging Face repository ID for remote saving
        **kwargs
            Additional arguments passed to save_to_hf
            
        Raises
        ------
        ValueError
            If path is None when repo_id is None
        """
        # if both path and repo_id are None, raise an error
        if path is None and repo_id is None:
            raise ValueError("path must be provided if repo_id is not given.")
        
        if repo_id is not None:
            self.save_to_hf(repo_id=repo_id, **kwargs)

        if path is not None:
            self.save_to_local(path)

    def load(self, path: Optional[str] = None, repo_id: Optional[str] = None, **kwargs) -> None:
        """Automatic load from either local disk or Hugging Face Hub.
        
        Parameters
        ----------
        path : Optional[str], default=None
            File path for local loading.
        repo_id : Optional[str], default=None
            Hugging Face repository ID for remote loading. If path is provided, repo_id is ignored.
        **kwargs
            Additional arguments passed to load_from_hf
            
        Raises
        ------
        FileNotFoundError
            If no local file is found and no repo_id is provided
        """
        if path is not None:
            if os.path.exists(path):
                self.load_from_local(path)
            else:
                raise FileNotFoundError(f"No local file found at '{path}'.")
        else:
            if repo_id is None:
                raise ValueError("repo_id must be provided if path is not given.")
            self.load_from_hf(repo_id, **kwargs)

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted.
        
        Raises
        ------
        AttributeError
            If the model is not fitted
        """
        if not self.is_fitted_:
            raise AttributeError("This model is not fitted yet. Call 'fit' before using it.")

    def __str__(self, N_CHAR_MAX: int = 700) -> str:
        """Return a string representation of the model.
        
        Parameters
        ----------
        N_CHAR_MAX : int, default=700
            Maximum number of characters in the string representation
            
        Returns
        -------
        str
            String representation of the model
        """
        attributes = {
            name: value
            for name, value in sorted(self.__dict__.items())
            if not name.startswith("_") and not callable(value)
        }
        attributes = {k: v for k, v in attributes.items() if k != "fitting_loss"}

        def format_value(v):
            """Helper to format values for representation."""
            if isinstance(v, (float, np.float32, np.float64)):
                return f"{v:.3g}"
            elif isinstance(v, (list, tuple, np.ndarray)) and len(v) > 6:
                return f"{v[:3]}...{v[-3:]}"
            elif isinstance(v, str) and len(v) > 50:
                return f"'{v[:25]}...{v[-22:]}'"
            elif isinstance(v, dict) and len(v) > 6:
                return f"{{{', '.join(f'{k}: {v}' for k, v in list(v.items())[:3])}...}}"
            elif isinstance(v, torch.nn.Module):
                return f"{v.__class__.__name__}(...)"
            return repr(v)

        class_name = self.__class__.__name__
        important_attrs = ["model_name", "is_fitted_", "task_type", "num_task"]
        attributes_str = [f"{attr}={format_value(attributes.pop(attr))}" for attr in important_attrs if attr in attributes]
        attributes_str += [f"{k}={format_value(v)}" for k, v in sorted(attributes.items())]

        content = ",\n    ".join(attributes_str)
        repr_str = f"{class_name}(\n    {content}\n)"
        
        if len(repr_str) > N_CHAR_MAX:
            repr_str = "\n".join([repr_str[:N_CHAR_MAX//2], "...", repr_str[-N_CHAR_MAX//2:]])
        
        return repr_str
    