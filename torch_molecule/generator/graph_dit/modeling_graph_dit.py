import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal, Type
import warnings
from dataclasses import dataclass, field

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .transformer import Transformer
from ...base import BaseMolecularGenerator
from ...utils import graph_from_smiles

@dataclass
class GraphDITMolecularGenerator(BaseMolecularGenerator):
    """This predictor implements the graph diffusion transformer for molecular generation.
    Paper: Graph Diffusion Transformers for Multi-Conditional Molecular Generation (https://openreview.net/forum?id=cfrDLD1wfO)
    Reference Code: https://github.com/liugangcode/Graph-DiT
    """
    # Model parameters
    generator_type: str = "transformer"
    num_layer: int = 6
    hidden_size: int = 1152
    dropout: float = 0.
    drop_condition: float = 0.
    num_heads: int = 16
    mlp_ratio: float = 4
    max_n_nodes: int = 50
    X_dim: int = 118
    E_dim: int = 5
    y_dim: int = 1
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 10000
    learning_rate: float = 0.0002
    grad_clip_value: Optional[float] = None
    weight_decay: float = 0.0
    weight_X = 1
    weight_E = 10
    
    # Scheduler parameters
    use_lr_scheduler: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # Sampling parameters
    guide_scale: float = 2.

    # Other parameters
    verbose: bool = False
    model_name: str = "GraphDITMolecularGenerator"
    model_class: Type[Transformer] = field(default=Transformer, init=False)

    # Non-init fields
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__post_init__()
        pass

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
            "generator_type",
            "num_layer",
            "hidden_size", 
            "dropout",
            "drop_condition",
            "num_heads",
            "mlp_ratio",
            "max_n_nodes",
            "X_dim",
            "E_dim",
            "y_dim",
            # Training Parameters
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "weight_X",
            "weight_E",
            "grad_clip_value",
            # Scheduler Parameters
            "use_lr_scheduler",
            "scheduler_factor", 
            "scheduler_patience",
            # Sampling Parameters
            "guide_scale",
            # Other Parameters
            "fitting_epoch",
            "fitting_loss",
            "device",
            "verbose",
            "model_name"
        ]
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
                
            hyperparameters = checkpoint["hyperparameters"]
            
            return {
                "num_layer": hyperparameters.get("num_layer", self.num_layer),
                "hidden_size": hyperparameters.get("hidden_size", self.hidden_size),
                "dropout": hyperparameters.get("dropout", self.dropout),
                "drop_condition": hyperparameters.get("drop_condition", self.drop_condition),
                "num_heads": hyperparameters.get("num_heads", self.num_heads),
                "mlp_ratio": hyperparameters.get("mlp_ratio", self.mlp_ratio),
                "max_n_nodes": hyperparameters.get("max_n_nodes", self.max_n_nodes),
                "X_dim": hyperparameters.get("X_dim", self.X_dim),
                "E_dim": hyperparameters.get("E_dim", self.E_dim),
                "y_dim": hyperparameters.get("y_dim", self.y_dim),
            }
        else:
            return {
                "num_layer": self.num_layer,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout,
                "drop_condition": self.drop_condition,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "guide_scale": self.guide_scale,
                "max_n_nodes": self.max_n_nodes,
                "X_dim": self.X_dim,
                "E_dim": self.E_dim,
                "y_dim": self.y_dim,
            }
        
    def _convert_to_pytorch_data(self, X, y=None):
        """Convert numpy arrays to PyTorch Geometric data format.
        """
        if self.verbose:
            iterator = tqdm(enumerate(X), desc="Converting molecules to graphs", total=len(X))
        else:
            iterator = enumerate(X)

        pyg_graph_list = []
        for idx, smiles_or_mol in iterator:
            if y is not None:
                properties = y[idx]
            else: 
                properties = None
            graph = graph_from_smiles(smiles_or_mol, properties)
            g = Data()
            g.num_nodes = graph["num_nodes"]
            g.edge_index = torch.from_numpy(graph["edge_index"])

            del graph["num_nodes"]
            del graph["edge_index"]

            if graph["edge_feat"] is not None:
                g.edge_attr = torch.from_numpy(graph["edge_feat"])
                del graph["edge_feat"]

            if graph["node_feat"] is not None:
                g.x = torch.from_numpy(graph["node_feat"])
                del graph["node_feat"]

            if graph["y"] is not None:
                g.y = torch.from_numpy(graph["y"])
                del graph["y"]
   
            if graph["morgan"] is not None:
                g.morgan = torch.tensor(graph["morgan"], dtype=torch.int8).view(1, -1)
                del graph["morgan"]
            
            if graph["maccs"] is not None:
                g.maccs = torch.tensor(graph["maccs"], dtype=torch.int8).view(1, -1)
                del graph["maccs"]

            pyg_graph_list.append(g)

        return pyg_graph_list
    
    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimization components including optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.grad_clip_value is not None:
            for group in optimizer.param_groups:
                group.setdefault("max_norm", self.grad_clip_value)
                group.setdefault("norm_type", 2.0)

        scheduler = None
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=1e-6,
                cooldown=0,
                eps=1e-8,
            )

        return optimizer, scheduler
    
    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]] = None,
    ) -> "GraphDITMolecularGenerator":
        """Fit the model to the training data with optional validation set.

        Parameters
        ----------
        X_train : List[str]
            Training set input molecular structures as SMILES strings
        y_train : Union[List, np.ndarray]
            Training set target values for representation learning
        Returns
        -------
        self : GraphDITMolecularGenerator
            Fitted estimator
        """
        raise NotImplementedError("Training not yet implemented for this model.")
        self.fitting_epoch = epoch
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, is_class):
        raise NotImplementedError("Training not yet implemented for this model.")


    def generate(self, n_samples: int, **kwargs) -> List[str]:
        """Generate molecular structures.

        Parameters
        ----------
        n_samples : int
            Number of molecules to generate
        **kwargs : dict
            Additional arguments for generation

        Returns
        -------
        List[str]
            Generated SMILES strings
        """
        raise NotImplementedError("Generation not yet implemented for this model.")