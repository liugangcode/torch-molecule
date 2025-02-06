from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, ClassVar, Union, List, Dict, Any, Tuple, Callable, Type, Literal

import torch
import numpy as np
from .base import BaseModel

@dataclass
class BaseMolecularGenerator(BaseModel, ABC):
    """Base class for molecular generation."""
    
    model_name: str = field(default="BaseMolecularGenerator")

    @abstractmethod
    def fit(self, X: List[str], y: Optional[np.ndarray] = None) -> "BaseMolecularGenerator":
        pass
    
    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> List[str]:
        """Generate molecular structures.
        """
        pass
    