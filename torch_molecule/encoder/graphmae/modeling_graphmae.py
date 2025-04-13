import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Tuple, List, Literal, Type

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .model import GNN
from .dataloader import DataLoaderMaskingPred
from ..constant import GNN_ENCODER_MODELS, GNN_ENCODER_READOUTS, GNN_ENCODER_PARAMS
from ...base import BaseMolecularEncoder
from ...utils import graph_from_smiles

ALLOWABLE_ENCODER_MODELS = GNN_ENCODER_MODELS
ALLOWABLE_ENCODER_READOUTS = GNN_ENCODER_READOUTS

@dataclass
class GraphMAEMolecularEncoder(BaseMolecularEncoder):
    """GraphMAE: Self-Supervised Masked Graph Autoencoders
    
    References
    ----------
    - Paper: https://arxiv.org/abs/2205.10803
    - Code: https://github.com/THUDM/GraphMAE/tree/main/chem
    
    Parameters
    ----------
    mask_rate : float, default=0.15
        Fraction of nodes to mask during training.
    mask_edge : bool, default=False
        Whether to mask edges in addition to nodes.
    predictor_type : str, default="gin"
        Type of predictor network to use for reconstruction.
        Options: ["gin", "gcn", "linear"]
    
    num_layer : int, default=5
        Number of message passing layers in the GNN.
    hidden_size : int, default=300
        Dimension of hidden node representations.
    drop_ratio : float, default=0.5
        Dropout probability.
    norm_layer : str, default="batch_norm"
        Type of normalization to use.
        Options: ["batch_norm", "layer_norm", "instance_norm", "graph_norm", "size_norm", "pair_norm"]
    
    encoder_type : str, default="gin-virtual"
        Type of GNN encoder to use.
        Options: ["gin-virtual", "gcn-virtual", "gin", "gcn"]
    readout : str, default="sum"
        Pooling method to use for graph-level representations.
        Options: ["sum", "mean", "max"]
    
    batch_size : int, default=128
        Batch size for training and inference.
    epochs : int, default=500
        Number of training epochs.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    grad_clip_value : Optional[float], default=None
        Maximum norm of gradients for gradient clipping. No clipping if None.
    weight_decay : float, default=0.0
        L2 regularization factor.
    
    use_lr_scheduler : bool, default=False
        Whether to use a learning rate scheduler.
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate when using scheduler.
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced.
    
    verbose : bool, default=False
        Whether to display progress bars and logs.
    model_name : str, default="GraphMAEMolecularEncoder"
        Name of the model.
    
    Examples
    --------
    >>> from torch_molecule import GraphMAEMolecularEncoder
    >>> encoder = GraphMAEMolecularEncoder(hidden_size=128, epochs=100)
    >>> encoder.fit(["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO", "C1=CC=CC=C1"])
    >>> representations = encoder.encode(["CCO"])
    """
    # Task related parameters
    mask_rate: float = 0.15
    mask_edge: bool = False # whether to mask edges
    predictor_type: str = "gin" # one of ["gin", "gcn", "linear"]

    # Model parameters
    num_layer: int = 5
    hidden_size: int = 300
    drop_ratio: float = 0.5
    norm_layer: str = "batch_norm" # one of ["batch_norm", "layer_norm", "instance_norm", "graph_norm", "size_norm", "pair_norm"]
    
    encoder_type: str = "gin-virtual" # one of ["gin-virtual", "gcn-virtual", "gin", "gcn"]
    readout: str = "sum" # one of ["sum", "mean", "max"]
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 500
    learning_rate: float = 0.001
    grad_clip_value: Optional[float] = None
    weight_decay: float = 0.0
    
    # Scheduler parameters
    use_lr_scheduler: bool = False
    scheduler_factor: float = 0.5 # if use_lr_scheduler is True
    scheduler_patience: int = 5 # if use_lr_scheduler is True
    
    # Other parameters
    verbose: bool = False
    model_name: str = "GraphMAEMolecularEncoder"
    
    # Non-init fields
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    model_class: Type[GNN] = field(default=GNN, init=False)
    
    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__post_init__()
        if self.encoder_type not in ALLOWABLE_ENCODER_MODELS:
            raise ValueError(f"Invalid encoder_model: {self.encoder_type}. Currently only {ALLOWABLE_ENCODER_MODELS} are supported.")
        if self.readout not in ALLOWABLE_ENCODER_READOUTS:
            raise ValueError(f"Invalid encoder_readout: {self.readout}. Currently only {ALLOWABLE_ENCODER_READOUTS} are supported.")

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        List[str]
            List of parameter names that can be used for model configuration.
        """
        return ["mask_rate", "mask_edge", "predictor_type"] + GNN_ENCODER_PARAMS.copy()
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = {
            "num_layer": self.num_layer,
            "hidden_size": self.hidden_size,
            "drop_ratio": self.drop_ratio,
            "norm_layer": self.norm_layer,
            "readout": self.readout,
            "encoder_type": self.encoder_type,
            "predictor_type": self.predictor_type,
            "mask_edge": self.mask_edge
        }
        
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            hyperparameters = checkpoint["hyperparameters"]
            params = {k: hyperparameters.get(k, v) for k, v in params.items()}
            
        return params
        
    def _convert_to_pytorch_data(self, X):
        """Convert numpy arrays to PyTorch Geometric data format.
        """
        if self.verbose:
            iterator = tqdm(enumerate(X), desc="Converting molecules to graphs", total=len(X))
        else:
            iterator = enumerate(X)

        pyg_graph_list = []
        for idx, smiles_or_mol in iterator:
            graph = graph_from_smiles(smiles_or_mol, None)
            g = Data()
            # g.num_nodes = graph["num_nodes"]
            g.edge_index = torch.from_numpy(graph["edge_index"])

            del graph["num_nodes"]
            del graph["edge_index"]

            if graph["edge_feat"] is not None:
                g.edge_attr = torch.from_numpy(graph["edge_feat"])
                del graph["edge_feat"]

            if graph["node_feat"] is not None:
                g.x = torch.from_numpy(graph["node_feat"])
                del graph["node_feat"]

            pyg_graph_list.append(g)

        return pyg_graph_list
    
    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimization components including optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
    ) -> "GraphMAEMolecularEncoder":
        """Fit the model to the training data with optional validation set.

        Parameters
        ----------
        X_train : List[str]
            Training set input molecular structures as SMILES strings
        Returns
        -------
        self : GraphMAEMolecularEncoder
            Fitted estimator
        """
        
        self._initialize_model(self.model_class)
        self.model.initialize_parameters()
        optimizer, scheduler = self._setup_optimizers()
        
        # Prepare datasets and loaders
        X_train, _ = self._validate_inputs(X_train, return_rdkit_mol=True)
        train_dataset = self._convert_to_pytorch_data(X_train)

        train_loader = DataLoaderMaskingPred(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, num_workers = 0, 
            mask_rate=self.mask_rate, 
            mask_edge=self.mask_edge)

        self.fitting_loss = []

        for epoch in range(self.epochs):
            # Training phase
            train_losses = self._train_epoch(train_loader, optimizer, epoch)
            self.fitting_loss.append(np.mean(train_losses))
            if scheduler:
                scheduler.step(np.mean(train_losses))

        self.fitting_epoch = epoch
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, epoch):
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
            loss_atom, loss_edge = self.model.compute_loss(batch)
            loss = loss_atom + loss_edge
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()
            losses.append(loss.item())

            if self.verbose:
                iterator.set_postfix({"Epoch": f"{epoch}", "Loss": f"{loss.item():.4f}", "Loss_atom": f"{loss_atom.item():.4f}", "Loss_edge": f"{loss_edge.item():.4f}"})

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