from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, ClassVar, Union, List, Dict, Any, Tuple, Callable, Type, Literal

import torch
import numpy as np
from .base import BaseModel

@dataclass
class BaseMolecularEncoder(BaseModel, ABC):
    """Base class for molecular representation learning."""
    
    model_name: str = field(default="BaseMolecularEncoder")

    @abstractmethod
    def encode(self, X: List[str], return_type: Literal["np", "pt"] = "pt") -> Union[np.ndarray, torch.Tensor]:
        pass
    
    @abstractmethod
    def fit(self, X: List[str], y: Optional[np.ndarray] = None) -> "BaseMolecularEncoder":
        pass

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