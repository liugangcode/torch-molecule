import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Literal

from dataclasses import dataclass

import torch

from ...base import BaseMolecularEncoder


@dataclass
class PretrainedMolecularEncoder(BaseMolecularEncoder):
    """This encoder uses a pretrained transformer model from HuggingFace."""
    # Task-related parameters
    # repo_id: str = "huggingface/PretrainedMolecularEncoder"
    model_name: str = "PretrainedMolecularEncoder"

    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__post_init__()
        self._require_transformers()
        self.is_fitted_ = True
        self.fitting_epoch = -1
        self.fitting_loss = -1

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        List[str]
            List of parameter names that can be used for model configuration.
        """
        return []

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = ["model_name"]
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            return {k: checkpoint["hyperparameters"][k] for k in params}
        return {k: getattr(self, k) for k in params}

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        raise NotImplementedError("PretrainedMolecularEncoder does not support training.")
    
    def save_to_local(self, path: str) -> None:
        raise NotImplementedError("PretrainedMolecularEncoder does not support saving to local.")
    
    def load_from_local(self, path: str) -> None:
        raise NotImplementedError("PretrainedMolecularEncoder does not support loading from local.")
    
    def save_to_hf(self) -> None:
        raise NotImplementedError("PretrainedMolecularEncoder does not support saving to huggingface.")
    
    def load_from_hf(self, repo_id: str) -> None:
        # TODO: Implement this
        raise NotImplementedError("Implements this.")
    
    def load(self, repo_id: str) -> None:
        self.load_from_hf(repo_id)

    def fit(self, repo_id: str) -> "PretrainedMolecularEncoder":
        self.load_from_hf(repo_id)
        return self

    def encode(self, X: List[str], return_type: Literal["np", "pt"] = "pt") -> Union[np.ndarray, torch.Tensor]:
        """Encode molecules into vector representations.

        Parameters
        ----------
        X : List[str]
            List of SMILES strings
        return_type : Literal["np", "pt"], default="pt"
            Return type of the representations

        Returns
        -------
        representations : ndarray or torch.Tensor
            Molecular representations
        """
        self._check_is_fitted()
        X, _ = self._validate_inputs(X, return_rdkit_mol=True)
        raise NotImplementedError("Implements this.")

        # Placeholder for transformer-based encodings
        # Replace with actual encoding logic when integrating the transformer model
        encodings = [X]  # dummy list to allow concat

        encodings = torch.cat(encodings, dim=0)
        return encodings if return_type == "pt" else encodings.numpy()

    @staticmethod
    def _require_transformers():
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for PretrainedMolecularEncoder. "
                "Please install it using `pip install transformers`."
            )