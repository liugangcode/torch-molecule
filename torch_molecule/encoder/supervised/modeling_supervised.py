import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Literal

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .model import GNN
from ..constant import GNN_ENCODER_MODELS, GNN_ENCODER_READOUTS, GNN_ENCODER_PARAMS
from ...base import BaseMolecularEncoder
from ...utils import graph_from_smiles
from ...utils import PSEUDOTASK

ALLOWABLE_ENCODER_MODELS = GNN_ENCODER_MODELS
ALLOWABLE_ENCODER_READOUTS = GNN_ENCODER_READOUTS

class SupervisedMolecularEncoder(BaseMolecularEncoder):
    """This encoder implements a GNN model for supervised molecular representation learning with user-defined or predefined fingerprint/calculated property tasks.

    Parameters
    ----------
    num_task : int, optional
        Number of user-defined tasks for supervised pretraining. If it is specified, user must provide y_train in the fit function.
    predefined_task : List[str], optional
        List of predefined tasks to use. Must be from the supported task list ["morgan", "maccs", "logP"]. If None and num_task is None, all predefined tasks will be used.
    encoder_type : str, default="gin-virtual" 
        Type of GNN architecture to use. One of ["gin-virtual", "gcn-virtual", "gin", "gcn"].
    readout : str, default="sum" 
        Method for aggregating node features to obtain graph-level representations. One of ["sum", "mean", "max"].
    num_layer : int, default=5
        Number of GNN layers.
    hidden_size : int, default=300
        Dimension of hidden node features.
    drop_ratio : float, default=0.5
        Dropout probability.
    norm_layer : str, default="batch_norm"
        Type of normalization layer to use. One of ["batch_norm", "layer_norm", "instance_norm", "graph_norm", "size_norm", "pair_norm"].
    batch_size : int, default=128
        Number of samples per batch for training.
    epochs : int, default=500
        Maximum number of training epochs.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    grad_clip_value : float, optional
        Maximum norm of gradients for gradient clipping.
    weight_decay : float, default=0.0
        L2 regularization strength.
    use_lr_scheduler : bool, default=False
        Whether to use a learning rate scheduler.
    scheduler_factor : float, default=0.5
        Factor by which to reduce the learning rate when plateau is detected.
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced.
    verbose : str, default="none"
        Whether to display progress info. Options are: "none", "progress_bar", "print_statement". If any other, "none" is automatically chosen.
    device : torch.device or str, optional
        Device to use for computation. Inherited from BaseMolecularEncoder.
    model_name : str, default="SupervisedMolecularEncoder"
        Name of the model. Inherited from BaseMolecularEncoder.
    """
    def __init__(
        self, 
        num_task: Optional[int] = None, 
        predefined_task: Optional[List[str]] = None, 
        encoder_type: str = "gin-virtual", 
        readout: str = "sum", 
        num_layer: int = 5, 
        hidden_size: int = 300, 
        drop_ratio: float = 0.5, 
        norm_layer: str = "batch_norm", 
        batch_size: int = 128, 
        epochs: int = 500, 
        learning_rate: float = 0.001, 
        grad_clip_value: Optional[float] = None, 
        weight_decay: float = 0.0, 
        use_lr_scheduler: bool = False, 
        scheduler_factor: float = 0.5, 
        scheduler_patience: int = 5, 
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "SupervisedMolecularEncoder",
        verbose: str = "none", 
    ):
        super().__init__(device=device, model_name=model_name, verbose=verbose)
        
        self.num_task = num_task
        self.predefined_task = predefined_task
        self.encoder_type = encoder_type
        self.readout = readout
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.drop_ratio = drop_ratio
        self.norm_layer = norm_layer
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.grad_clip_value = grad_clip_value
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.fitting_loss = list()
        self.fitting_epoch = 0
        self.model_class = GNN

        if self.encoder_type not in ALLOWABLE_ENCODER_MODELS:
            raise ValueError(f"Invalid encoder: {self.encoder_type}. Currently only {ALLOWABLE_ENCODER_MODELS} are supported.")
        if self.readout not in ALLOWABLE_ENCODER_READOUTS:
            raise ValueError(f"Invalid readout: {self.readout}. Currently only {ALLOWABLE_ENCODER_READOUTS} are supported.")
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
        return ["num_task", "predefined_task", "num_pretask"] + GNN_ENCODER_PARAMS.copy()
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = {
            "num_layer": self.num_layer,
            "hidden_size": self.hidden_size, 
            "num_task": self.num_task,
            "encoder_type": self.encoder_type,
            "drop_ratio": self.drop_ratio,
            "norm_layer": self.norm_layer,
            "readout": self.readout,
        }
        
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            hyperparameters = checkpoint["hyperparameters"]
            params = {k: hyperparameters.get(k, v) for k, v in params.items()}
            
        return params
    def _convert_to_pytorch_data(self, X, y=None):
        """Convert numpy arrays to PyTorch Geometric data format.
        """
        if self.verbose == "progress_bar":
            iterator = tqdm(enumerate(X), desc="Converting molecules to graphs", total=len(X))
        elif self.verbose == "print_statement":
            print("Converting molecules to graphs, preparing data for training...")
            iterator = enumerate(X)
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
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
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
    ) -> "SupervisedMolecularEncoder":
        """Fit the model to the training data with optional validation set.

        Parameters
        ----------
        X_train : List[str]
            Training set input molecular structures as SMILES strings
        y_train : Union[List, np.ndarray]
            Training set target values for representation learning
        Returns
        -------
        self : SupervisedMolecularEncoder
            Fitted estimator
        """
        user_defined_task = (self.num_task or 0) - self.num_pretask
        if user_defined_task > 0:
            if y_train is None:
                raise ValueError("User-defined tasks require target values but y_train is None.")
            y_train_arr = np.array(y_train) if isinstance(y_train, list) else y_train
            if y_train_arr.shape[1] != user_defined_task:
                raise ValueError(f"Number of user-defined tasks ({user_defined_task}) must match the number of target values in y_train ({y_train_arr.shape[1]}).")

        self._initialize_model(self.model_class)
        self.model.initialize_parameters()
        optimizer, scheduler = self._setup_optimizers()
        
        # Prepare datasets and loaders
        X_train, y_train = self._validate_inputs(X_train, y_train, return_rdkit_mol=True, num_task=self.num_task or 0, num_pretask=self.num_pretask)
        train_dataset = self._convert_to_pytorch_data(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        self.fitting_loss = []
        if user_defined_task > 0:
            is_class_user = self._inspect_task_types(y_train, return_type="pt")
        else:
            is_class_user = torch.tensor([], dtype=torch.bool)

        if self.predefined_task is not None:
            is_class_predefined = torch.cat([torch.full((PSEUDOTASK[task][0],), PSEUDOTASK[task][1] == "classification", dtype=torch.bool) for task in self.predefined_task])
        else:
            is_class_predefined = torch.tensor([], dtype=torch.bool)

        is_class = torch.cat([is_class_user, is_class_predefined])
        
        # Calculate total steps for global progress bar
        total_steps = self.epochs * len(train_loader)
        global_pbar = None
        if self.verbose == "progress_bar":  
            global_pbar = tqdm(total=total_steps, desc="Training Progress")
        
        for epoch in range(self.epochs):
            # Training phase
            train_losses = self._train_epoch(train_loader, optimizer, is_class, epoch, global_pbar)
            self.fitting_loss.append(float(np.mean(train_losses)))
            if scheduler:
                scheduler.step(float(np.mean(train_losses)))

        if global_pbar is not None:
            global_pbar.close()
        self.fitting_epoch = epoch
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, is_class, epoch, global_pbar):
        """Training logic for one epoch.

        Args:
            train_loader: DataLoader containing training data
            optimizer: Optimizer instance for model parameter updates
            is_class: Boolean tensor indicating classification tasks
            epoch: Current epoch number
            global_pbar: Global progress bar for all epochs

        Returns:
            list: List of loss values for each training step
        """
        self.model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            loss = self.model.compute_loss(batch, is_class)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()
            losses.append(loss.item())

            # Update global progress bar
            log_dict = {
                    "Epoch": f"{epoch + 1}/{self.epochs}",
                    "Loss": f"{loss.item():.4f}"
                }
            if self.verbose == "progress_bar" and global_pbar:
                global_pbar.set_postfix(log_dict)
                global_pbar.update(1)
            if self.verbose == "print_statement":
                print(log_dict)

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