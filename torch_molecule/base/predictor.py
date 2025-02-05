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
from ..utils.format import sanitize_config

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, ClassVar, Union, List, Dict, Any, Tuple, Callable, Type

@dataclass
class BaseMolecularPredictor(ABC):
    """Base class for molecular discovery estimators.

    This class provides a skeleton for implementing custom molecular discovery
    models following scikit-learn's conventions. It includes basic functionality
    for parameter management, fitting, prediction, and validation.

    Attributes
    ----------
    num_task : int
        Number of prediction tasks
    task_type : str
        Type of task ('classification' or 'regression')
    model_name : str
        Name of the model
    device : Optional[torch.device]
        Device to run the model on
    model : Optional[torch.nn.Module]
        Model instance
    """
    
    num_task: int
    task_type: str
    model_name: str = field(default="BaseMolecularPredictor")
    device: Optional[torch.device] = field(default=None)
    model: Optional[torch.nn.Module] = field(default=None, init=False)
    
    # Class variables
    DEFAULT_METRICS: ClassVar[Dict] = {
        "classification": {
            "default": ("roc_auc", roc_auc_score, True),  # (name, function, higher_better)
        },
        "regression": {"default": ("mae", mean_absolute_error, False)},
    }
    
    # Instance variables with default values
    is_fitted_: bool = field(default=False, init=False)
    model_class: Optional[Type[torch.nn.Module]] = field(default=None, init=False)
    
    def __post_init__(self):
        """Validate inputs and set device after initialization."""
        if self.task_type not in ["classification", "regression"]:
            raise ValueError(
                f"task_type must be one of ['classification', 'regression'], got {self.task_type}"
            )

        if not isinstance(self.num_task, (int, np.integer)) or self.num_task <= 0:
            raise ValueError(f"num_task must be a positive integer, got {self.num_task}")
            
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

    def set_params(self, **params) -> "BaseMolecularPredictor":
        """Set the parameters of this predictor.

        Parameters
        ----------
        **params : dict
            predictor parameters.

        Returns
        -------
        self : object
            predictor instance.
        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for predictor {self}")
            setattr(self, key, value)
        return self

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimization components including optimizer and learning rate scheduler.

        Returns
        -------
        Tuple[optim.Optimizer, Optional[Any]]
            A tuple containing:
            - The configured optimizer
            - The learning rate scheduler (if enabled, else None)
        """
        raise NotImplementedError("optimizers and schedulers must be implemented by child class")

    def _setup_evaluation(
        self,
        evaluate_criterion: Optional[Union[str, Callable]],
        evaluate_higher_better: Optional[bool],
    ) -> None:
        """Setup evaluation metric and direction.

        Parameters
        ----------
        evaluate_criterion : Union[str, Callable, None]
            Evaluation metric specification
        evaluate_higher_better : bool, optional
            Whether higher values are better
        """
        if evaluate_criterion is None:
            # Use default metric based on task type
            default_metric = self.DEFAULT_METRICS[self.task_type]["default"]
            self.evaluate_name = default_metric[0]
            self.evaluate_criterion = default_metric[1]
            self.evaluate_higher_better = default_metric[2]

        else:
            if isinstance(evaluate_criterion, str):
                # Map string to predefined metric
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
                # Custom metric function
                if evaluate_higher_better is None:
                    raise ValueError(
                        "evaluate_higher_better must be specified "
                        "when using a custom evaluation function"
                    )

                self.evaluate_name = "custom"
                self.evaluate_criterion = evaluate_criterion
                self.evaluate_higher_better = evaluate_higher_better

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, X_unlbl=None) -> "BaseMolecularPredictor":
        """Fit the model to the data.

        Parameters
        ----------
        X_train / X_val / X_unlbl: list of SMILES
        y_train / y_val: array-like of shape (n_samples,), optional
            Target values.
        Returns
        -------
        self : object
            Fitted estimator.
        """
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
        """AutoFit the model to the data.

        Parameters
        ----------
        X_train / X_val : list of SMILES
        y_train / y_val: array-like of shape (n_samples,), optional
            Target values.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        pass

    def _train_epoch(self, train_loader, optimizer):
        """Training logic for one epoch.

        Should be implemented by child classes based on specific model architecture.
        """
        raise NotImplementedError("Training epoch logic must be implemented by child class")

    def _evaluation_epoch(self, evaluate_loader):
        """Training logic for one evaluation.

        Should be implemented by child classes based on specific model architecture.
        """
        raise NotImplementedError("Evaluation epoch logic must be implemented by child class")

    def _check_is_fitted(self) -> None:
        """Check if the estimator is fitted.

        Raises
        ------
        NotFittedError
            If the estimator is not fitted.
        """
        if not self.is_fitted_:
            raise AttributeError(
                "This MolecularBaseEstimator instance is not fitted yet. "
                "Call 'fit' before using this estimator."
            )

    def __repr__(self, N_CHAR_MAX=700):
        """Get string representation of the estimator.

        Parameters
        ----------
        N_CHAR_MAX : int, default=700
            Maximum number of non-blank characters to show

        Returns
        -------
        str
            String representation of the estimator
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
        important_attrs = ["num_task", "task_type", "model_name", "is_fitted_"]
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
        """Get model parameters either from checkpoint or current instance.

        Parameters
        ----------
        checkpoint : Optional[Dict]
            Checkpoint containing model hyperparameters

        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters

        Raises
        ------
        NotImplementedError
            If the child class doesn't implement this method
        ValueError
            If checkpoint contains invalid parameters
        """
        raise NotImplementedError(
            "Method _get_model_params must be implemented by child class. "
            "This method should return a dictionary of model parameters either "
            "from a checkpoint or current instance values."
        )

    def _initialize_model(
        self, model_class: Type[torch.nn.Module], device, checkpoint: Optional[Dict] = None
    ) -> None:
        """Initialize the GNN model with given parameters or checkpoint.

        Parameters
        ----------
        checkpoint : Optional[Dict]
            If provided, should contain:
                - hyperparameters: Dict of model configuration
                - model_state_dict: Dict of model state
            If None, initializes model with current instance parameters.

        Raises
        ------
        ValueError
            If checkpoint is provided but missing required keys or contains invalid parameters
        RuntimeError
            If device initialization fails
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

    def _load_default_criterion(self):
        """Sets default criterion based on task_type."""
        if self.task_type == "regression":
            return torch.nn.L1Loss()
        elif self.task_type == "classification":
            return torch.nn.BCEWithLogitsLoss()
        else:
            warnings.warn(
                "Unknown task type. Using MSE Loss as the default criterion. "
                "Please specify a valid task type ('regression' or 'classification') or provide a criterion."
            )
            return torch.nn.L1Loss()

    def _validate_inputs(
        self, X: List[str], y: Optional[Union[List, np.ndarray]] = None, return_rdkit_mol: bool = True
    ) -> Tuple[Union[List[str], List["Mol"]], Optional[np.ndarray]]:
        """Validate input SMILES strings and target values.

        Parameters
        ----------
        X : List[str]
            List of SMILES strings
        y : Union[List, np.ndarray], optional
            Target values. Can be:
            - 1D array/list for single task (num_task must be 1)
            - 2D array/list for multiple tasks
        return_rdkit_mol : bool, default=False
            If True, returns RDKit Mol objects instead of SMILES strings

        Returns
        -------
        X : Union[List[str], List[Mol]]
            Validated SMILES strings or RDKit Mol objects if return_rdkit_mol=True
        y : np.ndarray or None
            Validated target values as numpy array, reshaped to 2D if needed

        Raises
        ------
        ValueError
            If inputs don't meet requirements
        """
        # Validate X
        if not isinstance(X, list):
            raise ValueError("X must be a list of SMILES strings")

        if not all(isinstance(s, str) for s in X):
            raise ValueError("All elements in X must be strings")

        # Validate SMILES using RDKit and store Mol objects
        invalid_smiles = []
        rdkit_mols = []
        for i, smiles in enumerate(X):
            is_valid, error_msg, mol = self._validate_smiles(smiles, i)
            if not is_valid:
                invalid_smiles.append(error_msg)
            else:
                rdkit_mols.append(mol)

        if invalid_smiles:
            raise ValueError("Invalid SMILES found:\n" + "\n".join(invalid_smiles))

        # Validate y if provided
        if y is not None:
            try:
                # Convert to numpy array if it's a list
                y = np.asarray(y, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Could not convert y to numpy array: {str(e)}")

            # Handle 1D array case
            if len(y.shape) == 1:
                if self.num_task != 1:
                    raise ValueError(
                        f"1D target array provided but num_task is {self.num_task}. "
                        "For multiple tasks, y must be 2D array."
                    )
                # Reshape to 2D
                y = y.reshape(-1, 1)

            # Validate dimensions
            if len(y.shape) != 2:
                raise ValueError(
                    f"y must be 1D (for single task) or 2D (for multiple tasks), "
                    f"got shape {y.shape}"
                )

            # Check sample dimension
            if y.shape[0] != len(X):
                raise ValueError(
                    f"First dimension of y ({y.shape[0]}) must match " f"length of X ({len(X)})"
                )

            # Check task dimension
            if y.shape[1] != self.num_task:
                raise ValueError(
                    f"Number of tasks in y ({y.shape[1]}) must match "
                    f"num_task ({self.num_task})"
                )

            # Handle infinite values
            inf_mask = ~np.isfinite(y)
            if np.any(inf_mask):
                inf_indices = np.where(inf_mask)
                warnings.warn(
                    f"Infinite values found in y at indices: {list(zip(*inf_indices))}. "
                    "Converting to NaN.",
                    RuntimeWarning,
                )
                y = y.astype(float)  # Ensure float type for NaN support
                y[inf_mask] = np.nan

            # Check for NaN values after conversion
            nan_mask = np.isnan(y)
            if np.any(nan_mask):
                # Count NaNs per task
                nan_counts = np.sum(nan_mask, axis=0)
                nan_percentages = (nan_counts / len(X)) * 100

                # Create detailed warning message
                task_warnings = []
                for task_idx, (count, percentage) in enumerate(zip(nan_counts, nan_percentages)):
                    if count > 0:
                        task_warnings.append(f"Task {task_idx}: {count} NaNs ({percentage:.1f}%)")

                warnings.warn(
                    "NaN values present in y:\n"
                    + "\n".join(task_warnings)
                    + "\nSamples with NaN values may be ignored during training.",
                    RuntimeWarning,
                )

        return rdkit_mols if return_rdkit_mol else X, y

    def _validate_smiles(self, smiles: str, idx: int) -> Tuple[bool, Optional[str], Optional["Mol"]]:
        """Validate a single SMILES string using RDKit.

        Parameters
        ----------
        smiles : str
            SMILES string to validate
        idx : int
            Index of the SMILES in the input list (for error reporting)

        Returns
        -------
        is_valid : bool
            Whether the SMILES is valid
        error_msg : str or None
            Error message if invalid, None if valid
        mol : Mol or None
            RDKit Mol object if valid, None if invalid
        """
        if not smiles or not smiles.strip():
            return False, f"Empty SMILES at index {idx}", None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, f"Invalid SMILES structure at index {idx}: {smiles}", None
            return True, None, mol
        except Exception as e:
            return False, f"RDKit error at index {idx}: {str(e)}", None

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator."""
        return ["num_task", "task_type", "is_fitted_"]

    def save_model(self, path: str) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : str
            Path where to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")

        if not path.endswith((".pt", ".pth")):
            raise ValueError("Save path should end with '.pt' or '.pth'")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        model_name = os.path.splitext(os.path.basename(path))[0]
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "hyperparameters": self.get_params(),
            "model_name": model_name,
            "date_saved": datetime.datetime.now().isoformat(),
            "version": getattr(self, "__version__", "1.0.0"),
        }
        torch.save(save_dict, path)

    def load_model(self, path: str, repo_id: Optional[str] = None):
        """Load a saved model from disk or download from HuggingFace hub.

        Parameters
        ----------
        path : str
            Path to the saved model file or desired local path for downloaded model
        repo_id : str, optional
            HuggingFace model repository ID (e.g., 'username/model-name')
            If provided and local path doesn't exist, will attempt to download from hub

        Returns
        -------
        self : GNNMolecularPredictor
            Updated model instance with loaded weights and parameters

        Raises
        ------
        FileNotFoundError
            If the model file doesn't exist locally and no repo is provided,
            or if download from repo fails
        ValueError
            If the saved file is corrupted or incompatible
        """
        # First try to load from local path
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device)
            except Exception as e:
                raise ValueError(f"Error loading model from {path}: {str(e)}")

        # If local file doesn't exist and repo is provided, try downloading
        elif repo_id is not None:
            try:
                from huggingface_hub import hf_hub_download

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

                # Download from HuggingFace using specified parameters
                model_name = os.path.splitext(os.path.basename(path))[0]

                # Download model file
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{model_name}.pt",
                    local_dir=os.path.dirname(path),
                    # local_dir_use_symlinks=False
                )
                hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                    local_dir=os.path.dirname(path),
                )
                # Load the downloaded model
                checkpoint = torch.load(downloaded_path, map_location=self.device)

            except Exception as e:
                if os.path.exists(path):
                    os.remove(path)  # Clean up partial downloads
                raise FileNotFoundError(
                    f"Failed to download model from repository '{repo_id}': {str(e)}"
                )

        # If neither local file exists nor repo provided
        else:
            raise FileNotFoundError(f"No model file found at '{path}' and no repository provided")

        try:
            # Validate checkpoint contents
            required_keys = {
                "model_state_dict",
                "hyperparameters",
                "model_name",
            }
            if not all(key in checkpoint for key in required_keys):
                missing_keys = required_keys - set(checkpoint.keys())
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            # Validate and track all hyperparameter comparisons
            parameter_status = []
            for key, new_value in checkpoint["hyperparameters"].items():
                if hasattr(self, key):
                    old_value = getattr(self, key)
                    is_changed = old_value != new_value
                    parameter_status.append(
                        {
                            "Parameter": key,
                            "Old Value": old_value,
                            "New Value": new_value,
                            "Status": "Changed" if is_changed else "Unchanged",
                        }
                    )
                    if is_changed:
                        setattr(self, key, new_value)

            # Print comprehensive parameter status table
            if parameter_status:
                print("\nHyperparameter Status:")
                print("-" * 80)
                print(f"{'Parameter':<20} {'Old Value':<20} {'New Value':<20} {'Status':<10}")
                print("-" * 80)

                # Sort by status (Changed first, then Unchanged)
                parameter_status.sort(key=lambda x: (x["Status"] != "Changed", x["Parameter"]))

                for param in parameter_status:
                    status_color = (
                        "\033[91m" if param["Status"] == "Changed" else "\033[92m"
                    )  # Red for changed, green for unchanged
                    print(
                        f"{param['Parameter']:<20} "
                        f"{str(param['Old Value']):<20} "
                        f"{str(param['New Value']):<20} "
                        f"{status_color}{param['Status']}\033[0m"
                    )
                print("-" * 80)

                # Print summary
                changes_count = sum(1 for p in parameter_status if p["Status"] == "Changed")
                print(
                    f"\nSummary: {changes_count} parameters changed, "
                    f"{len(parameter_status) - changes_count} unchanged"
                )

            self._initialize_model(self.model_class, self.device, checkpoint)
            self.model_name = checkpoint["model_name"]
            self.is_fitted_ = True

            # Move model to correct device
            self.model = self.model.to(self.device)

            print(f"Model successfully loaded from {'repository' if repo_id else 'local path'}")
            return self

        except Exception as e:
            raise ValueError(f"Error loading model checkpoint: {str(e)}")

        finally:
            # Clean up any temporary files
            if (
                repo_id is not None
                and os.path.exists(downloaded_path)
                and os.path.abspath(downloaded_path) != os.path.abspath(path)
            ):
                try:
                    os.remove(downloaded_path)
                except Exception:
                    pass

    def push_to_huggingface(
        self,
        repo_id: str,
        task_id: str = "default",
        metrics: Optional[Dict[str, float]] = None,
        commit_message: str = "Update model",
        token: Optional[str] = None,
        private: bool = False,
    ) -> None:
        """Push a task-specific model checkpoint to Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The ID of the model repository (format: '<username>/<model_name>')
        task_id : str
            Task identifier (e.g., 'O2', 'N2')
        metrics : Optional[Dict[str, float]]
            Optional dictionary of metric names and values for this task
        commit_message : str, default="Update model"
            Message to commit with
        token : Optional[str], default=None
            HuggingFace token
        private : bool, default=False
            Whether to make the repository private
        """
        try:
            from huggingface_hub import HfApi, create_repo, metadata_update
            from .utils.hf_hub import merge_task_configs, get_existing_repo_data, create_model_card
        except ImportError:
            raise ImportError(
                "huggingface_hub package is required to push to Hugging Face Hub. "
                "Install it with: pip install huggingface_hub"
            )

        # Validate inputs
        if not isinstance(repo_id, str) or "/" not in repo_id:
            raise ValueError("repo_id must be in format '<username>/<model_name>'")
        if not task_id:
            raise ValueError("task_id must be provided")

        self._check_is_fitted()

        try:
            # Initialize API and check existing repo
            api = HfApi(token=token)
            repo_exists, existing_config, existing_readme = get_existing_repo_data(repo_id, token)

            # Create repo if it doesn't exist
            create_repo(repo_id, private=private, token=token, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save task-specific model checkpoint
                model_path = os.path.join(tmp_dir, f"{self.model_name}.pt")
                self.save_model(model_path)

                # Prepare task-specific configuration
                task_config = {
                    "task_type": self.task_type,
                    "config": sanitize_config(self.get_params(deep=True)),
                    "metrics": metrics or {},
                }

                # Create or update config
                base_config = (
                    {
                        "model_name": self.model_name,
                        "framework": "torch_molecule",
                        "date_created": existing_config.get(
                            "date_created", datetime.datetime.now().isoformat()
                        ),
                    }
                    if not repo_exists
                    else existing_config
                )

                # Merge task configuration
                final_config = merge_task_configs(
                    task_id=task_id,
                    existing_config=base_config,
                    new_task_config=task_config,
                    num_params=sum(p.numel() for p in self.model.parameters()),
                )

                # Save config
                config_path = os.path.join(tmp_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(final_config, f, indent=2)

                # Create and save README
                readme_content = create_model_card(
                    model_class=self.__class__.__name__,
                    model_name=self.model_name,
                    tasks_config=final_config.get("tasks", {}),
                    model_config=final_config,
                    repo_id=repo_id,
                    existing_readme=existing_readme,
                )

                readme_path = os.path.join(tmp_dir, "README.md")
                with open(readme_path, "w") as f:
                    f.write(readme_content)

                # Push files to hub
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=tmp_dir,
                    commit_message=f"{commit_message} - Task: {task_id}",
                )

                # Update repository metadata
                metadata = {
                    "tags": [
                        "torch_molecule",
                        "molecular-property-prediction",
                    ],
                    "library_name": "torch_molecule",
                    "pipeline_tag": "graph-ml",
                }
                metadata_update(repo_id=repo_id, metadata=metadata, token=token, overwrite=True)

                task_config = final_config["tasks"][task_id]
                print(f"Successfully pushed model for task {task_id} to {repo_id}")
                print(f"Task version: {task_config['current_version']}")
                if metrics:
                    print("Metrics:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")

        except Exception as e:
            raise RuntimeError(f"Failed to push to Hugging Face Hub: {str(e)}")