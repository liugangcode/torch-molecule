import os
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal, Type
import warnings
from dataclasses import dataclass, field

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .GNNModel import GNN
from ...base import BaseMolecularEncoder
from ...utils import graph_from_smiles
from ...utils import PSEUDOTASK
from ...utils.search import (
    suggest_parameter,
    ParameterSpec,
    ParameterType,
    parse_list_params,
)

ALLOWABLE_ENCODER_MODELS = ["gin-virtual", "gcn-virtual", "gin", "gcn"]
ALLOWABLE_ENCODER_READOUTS = ["sum", "mean", "max"]

@dataclass
class MoamaMolecularEncoder(BaseMolecularEncoder):
    """This encoder implements a GNN model for molecular representation learning.
    """
    # pretraining task
    num_task: Optional[int] = None
    num_pretask: Optional[int] = None
    predefined_task: Optional[List[str]] = None
    task_type: str = "classification"
    # Model parameters
    num_layer: int = 5
    hidden_size: int = 300
    drop_ratio: float = 0.5
    norm_layer: str = "batch_norm"
    
    encoder_model: str = "gin-virtual"
    encoder_readout: str = "sum"
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 500
    learning_rate: float = 0.001
    grad_clip_value: Optional[float] = None
    weight_decay: float = 0.0
    
    # Scheduler parameters
    use_lr_scheduler: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # Other parameters
    verbose: bool = False
    model_name: str = "MoamaMolecularEncoder"
    
    # Non-init fields
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    model_class: Type[GNN] = field(default=GNN, init=False)
    
    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__post_init__()
        if self.encoder_model not in ALLOWABLE_ENCODER_MODELS:
            raise ValueError(f"Invalid encoder_model: {self.encoder_model}. Currently only {ALLOWABLE_ENCODER_MODELS} are supported.")
        if self.encoder_readout not in ALLOWABLE_ENCODER_READOUTS:
            raise ValueError(f"Invalid encoder_readout: {self.encoder_readout}. Currently only {ALLOWABLE_ENCODER_READOUTS} are supported.")
        if self.predefined_task is not None:
            for task in self.predefined_task:
                if task not in PSEUDOTASK.keys():
                    raise ValueError(f"Invalid predefined_task: {task}. Currently only {PSEUDOTASK.keys()} are supported.")
        
        # Calculate number of predefined tasks if any are specified
        num_pretask = 0
        if self.predefined_task is not None:
            num_pretask = sum(PSEUDOTASK[task][0] for task in self.predefined_task)
        elif self.predefined_task is None and self.num_task is None:
            # Use all predefined tasks if none specified
            self.predefined_task = list(PSEUDOTASK.keys())
            num_pretask = sum(task[0] for task in PSEUDOTASK.values())

        self.num_pretask = num_pretask
        self.num_task = (self.num_task or 0) + num_pretask

        if self.verbose:
            if self.predefined_task is None:
                print(f"Using {self.num_task} user-defined tasks.")
            elif self.num_task == num_pretask:
                print(f"Using {num_pretask} predefined tasks from: {self.predefined_task}")
            else:
                print(f"Using {num_pretask} predefined tasks and {self.num_task - num_pretask} user-defined tasks.")

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        List[str]
            List of parameter names that can be used for model configuration.
        """
        return [
            # Task Parameters
            "num_task",
            "predefined_task",
            "num_pretask",
            # Model Hyperparameters
            "num_layer",
            "hidden_size", 
            "drop_ratio",
            "norm_layer",
            "encoder_model",
            "encoder_readout",
            # Training Parameters
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "grad_clip_value",
            # Scheduler Parameters
            "use_lr_scheduler",
            "scheduler_factor", 
            "scheduler_patience",
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
                "num_task": hyperparameters.get("num_task", self.num_task),
                "drop_ratio": hyperparameters.get("drop_ratio", self.drop_ratio),
                "norm_layer": hyperparameters.get("norm_layer", self.norm_layer),
                "encoder_readout": hyperparameters.get("encoder_readout", self.encoder_readout),
                "encoder_model": hyperparameters.get("encoder_model", self.encoder_model),
            }
        else:
            return {
                "num_layer": self.num_layer,
                "hidden_size": self.hidden_size,
                "num_task": self.num_task,
                "encoder_model": self.encoder_model,
                "drop_ratio": self.drop_ratio,
                "norm_layer": self.norm_layer,
                "encoder_readout": self.encoder_readout,
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
            graph = graph_from_smiles(smiles_or_mol, properties, augmented_properties = self.predefined_task)
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
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
    ) -> "MoamaMolecularEncoder":
        """Fit the model to the training data with optional validation set.

        Parameters
        ----------
        X_train : List[str]
            Training set input molecular structures as SMILES strings
        y_train : Union[List, np.ndarray]
            Training set target values for representation learning
        Returns
        -------
        self : MoamaMolecularEncoder
            Fitted estimator
        """

        self._initialize_model(self.model_class)
        self.model.initialize_parameters()
        optimizer, scheduler = self._setup_optimizers()
        
        # Prepare datasets and loaders
        smiles = X_train
        X_train, y_train = self._validate_inputs(X_train, y_train, return_rdkit_mol=True, num_task=self.num_task, num_pretask=self.num_pretask)
        train_dataset = self._convert_to_pytorch_data(X_train, y_train)
        for i in range(len(X_train)):
            train_dataset[i].smiles = smiles[i]
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        self.fitting_loss = []

        for epoch in range(self.epochs):
            # Training phase
            train_losses = self._train_epoch(train_loader, optimizer)
            self.fitting_loss.append(np.mean(train_losses))
            if scheduler:
                scheduler.step(np.mean(train_losses))

        self.fitting_epoch = epoch
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer):
        """Training logic for one epoch.

        Args:
            train_loader: DataLoader containing training data
            optimizer: Optimizer instance for model parameter updates

        Returns:
            list: List of loss values for each training step
        """
        self.model.train()
        losses = []

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )

        for batch in iterator:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()
            losses.append(loss.item())

            # Update progress bar if using tqdm
            if self.verbose:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        return losses

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

        # Convert to PyTorch Geometric format and create loader
        X, _ = self._validate_inputs(X, return_rdkit_mol=True)
        dataset = self._convert_to_pytorch_data(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Generate encodings
        self.model = self.model.to(self.device)
        self.model.eval()
        encodings = []
        with torch.no_grad():
            for batch in tqdm(loader, disable=not self.verbose):
                batch = batch.to(self.device)
                out = self.model(batch)
                encodings.append(out["graph"].cpu())

        # Concatenate and convert to requested format
        encodings = torch.cat(encodings, dim=0)
        return encodings if return_type == "pt" else encodings.numpy()