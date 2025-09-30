from abc import ABC, abstractmethod
from typing import Optional, List, Union
import torch
import numpy as np
from .base import BaseModel

class BaseMolecularGenerator(BaseModel, ABC):
    """Base class for molecular generation."""
    def __init__(
        self,
        *,
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "BaseMolecularGenerator",
        verbose: str = "none",
    ):
        super().__init__(device=device, model_name=model_name, verbose=verbose)

    @abstractmethod
    def fit(self, X: List[str], y: Optional[np.ndarray] = None) -> "BaseMolecularGenerator":
        pass
    
    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> List[str]:
        """Generate molecular structures.
        """
        pass