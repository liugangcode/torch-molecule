import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Type
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .sde import load_sde
from .solver import get_pc_sampler
from .model import GDSSModel, get_sde_loss_fn
from .utils import compute_dataset_info, to_dense, quantize_mol

from ...base import BaseMolecularGenerator
from ...utils import graph_from_smiles, graph_to_smiles

@dataclass
class GDSSMolecularGenerator(BaseMolecularGenerator):
    """ This generator implements "Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations"

    References
    ----------
    - Paper: https://arxiv.org/abs/2202.02514
    - Official Implementation: https://github.com/harryjo97/GDSS

    
    Parameters
    ----------
    num_layer : int, default=3
        Number of layers in the score networks.
    hidden_size_adj : float, default=8
        Hidden dimension size for the adjacency in the adjacency score network.
    hidden_size : int, default=16
        Hidden dimension size latent representation.
    attention_dim : int, default=16
        Dimension of attention layers.
    num_head : int, default=4
        Number of attention heads.
    sde_type_x : str, default='VE'
        SDE type for node features. One of 'VP', 'VE', 'subVP'.
    sde_beta_min_x : float, default=0.1
        Minimum noise level for node features.
    sde_beta_max_x : float, default=1
        Maximum noise level for node features.
    sde_num_scales_x : int, default=1000
        Number of noise scales for node features.
    sde_type_adj : str, default='VE'
        SDE type for adjacency matrix. One of 'VP', 'VE', 'subVP'.
    sde_beta_min_adj : float, default=0.1
        Minimum noise level for adjacency matrix.
    sde_beta_max_adj : float, default=1
        Maximum noise level for adjacency matrix.
    sde_num_scales_adj : int, default=1000
        Number of noise scales for adjacency matrix.
    batch_size : int, default=128
        Batch size for training.
    epochs : int, default=500
        Number of training epochs.
    learning_rate : float, default=0.005
        Learning rate for optimizer.
    grad_clip_value : Optional[float], default=1
        Value for gradient clipping. None means no clipping.
    weight_decay : float, default=1e-4
        Weight decay for optimizer.
    use_loss_reduce_mean : bool, default=False
        Whether to use mean reduction for loss calculation.
    use_lr_scheduler : bool, default=False
        Whether to use learning rate scheduler.
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate when using scheduler (only used if use_lr_scheduler is True).
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced (only used if use_lr_scheduler is True).
    sampler_predictor : str, default='Reverse'
        Predictor method for sampling. One of 'Euler', 'Reverse'.
    sampler_corrector : str, default='Langevin'
        Corrector method for sampling. One of 'Langevin', 'None'.
    sampler_snr : float, default=0.2
        Signal-to-noise ratio for corrector.
    sampler_scale_eps : float, default=0.7
        Scale factor for noise level in corrector.
    sampler_n_steps : int, default=1
        Number of corrector steps per predictor step.
    sampler_probability_flow : bool, default=False
        Whether to use probability flow ODE for sampling.
    sampler_noise_removal : bool, default=True
        Whether to remove noise in the final step of sampling.
    verbose : bool, default=False
        Whether to display progress bars and logs.
        
    """
    
    # Model parameters
    num_layer: int = 3
    hidden_size_adj: float = 8
    hidden_size: int = 16
    attention_dim: int = 16
    num_head: int = 4

    # Diffusion parameters
    sde_type_x: str = 'VE' # One of 'VP', 'VE', 'subVP'
    sde_beta_min_x: float = 0.1
    sde_beta_max_x: float = 1
    sde_num_scales_x: int = 1000
    sde_type_adj: str = 'VE' # One of 'VP', 'VE', 'subVP'
    sde_beta_min_adj: float = 0.1
    sde_beta_max_adj: float = 1
    sde_num_scales_adj: int = 1000

    # Training parameters
    batch_size: int = 128
    epochs: int = 500
    learning_rate: float = 0.005
    grad_clip_value: Optional[float] = 1.0
    weight_decay: float = 1e-4
    use_loss_reduce_mean: bool = False

    # Scheduler parameters
    use_lr_scheduler: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Sampling parameters
    sampler_predictor: str = 'Reverse'  # One of 'Euler', 'Reverse', others will be treated as 'Euler'
    sampler_corrector: str = 'Langevin' # One of 'Langevin', 'None', others will be treated as 'None'
    sampler_snr: float = 0.2
    sampler_scale_eps: float = 0.7
    sampler_n_steps: int = 1
    sampler_probability_flow: bool = False
    sampler_noise_removal: bool = True

    verbose: bool = False
    # Attributes
    model_name: str = "GDSSMolecularGenerator"
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    model_class: Type[GDSSModel] = field(default=GDSSModel, init=False)
    # dataset_info: Dict[str, Any] = field(default_factory=dict, init=False)
    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__post_init__()
        self.max_node = None
        self.input_dim_X = None
        self.input_dim_adj = None
        self.conv = 'GCN'
        self.sampler_eps = 1e-4
        self.train_eps = 1e-5
        self.dataset_info = None
        self.loss_fn = None
        self.sampling_fn = None

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
            "max_node", "num_layer", "num_head", "input_dim_X", "input_dim_adj", "hidden_size_adj", "hidden_size", "attention_dim",
            # Diffusion parameters  
            "dataset_info", "sde_type_x", "sde_beta_min_x", "sde_beta_max_x", "sde_num_scales_x", 
            "sde_type_adj", "sde_beta_min_adj", "sde_beta_max_adj", "sde_num_scales_adj",
            # Training Parameters
            "batch_size", "epochs", "learning_rate", "grad_clip_value", 
            "weight_decay", "use_loss_reduce_mean", "train_eps", "conv",
            # Scheduler Parameters
            "use_lr_scheduler", "scheduler_factor", "scheduler_patience",
            # Sampling Parameters
            "sampler_predictor", "sampler_corrector", "sampler_snr", "sampler_scale_eps", 
            "sampler_n_steps", "sampler_probability_flow", "sampler_noise_removal", "sampler_eps",
            # Other Parameters
            "fitting_epoch", "fitting_loss", "device", "verbose", "model_name"
        ]
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = ["input_dim_X", "max_node", "hidden_size", "num_layer", "input_dim_adj", "hidden_size_adj", "attention_dim", "num_head"]

        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            return {k: checkpoint["hyperparameters"][k] for k in params}
        
        return {k: getattr(self, k) for k in params}
        
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
            
            # No H, first heavy atom has type 0
            node_type = torch.from_numpy(graph['node_feat'][:, 0] - 1)
            
            # Filter out invalid node types (< 0)
            valid_mask = node_type >= 0
            if not valid_mask.all():
                # Get valid nodes and adjust edge indices
                valid_indices = torch.where(valid_mask)[0]
                index_map = -torch.ones(node_type.size(0), dtype=torch.long)
                index_map[valid_indices] = torch.arange(valid_indices.size(0))
                
                # Filter edges that connect to invalid nodes
                edge_index = torch.from_numpy(graph["edge_index"])
                valid_edges_mask = valid_mask[edge_index[0]] & valid_mask[edge_index[1]]
                valid_edge_index = edge_index[:, valid_edges_mask]
                
                # Remap edge indices to account for removed nodes
                remapped_edge_index = index_map[valid_edge_index]
                
                # Filter edge attributes
                edge_attr = torch.from_numpy(graph["edge_feat"])[:, 0] + 1
                valid_edge_attr = edge_attr[valid_edges_mask]
                
                # Update node and edge data
                node_type = node_type[valid_mask]
                g.edge_index = remapped_edge_index
                g.edge_attr = valid_edge_attr.long().squeeze(-1)
            else:
                # No invalid nodes, proceed normally
                g.edge_index = torch.from_numpy(graph["edge_index"])
                edge_attr = torch.from_numpy(graph["edge_feat"])[:, 0] + 1
                g.edge_attr = edge_attr.long().squeeze(-1)
            
            # * is encoded as "misc" which is 119 - 1 and should be 117
            node_type[node_type == 118] = 117
            g.x = node_type.long().squeeze(-1)
            del graph["node_feat"]
            del graph["edge_index"]
            del graph["edge_feat"]

            g.y = torch.from_numpy(graph["y"])
            del graph["y"]

            pyg_graph_list.append(g)

        return pyg_graph_list

    def _setup_diffusion_params(self, X: Union[List, Dict]) -> None:
        # Extract dataset info from X if it's a dict (from checkpoint), otherwise compute it
        if isinstance(X, dict):
            dataset_info = X["hyperparameters"]["dataset_info"]
            max_node = X["hyperparameters"]["max_node"]
        else:
            assert isinstance(X, list)
            dataset_info = compute_dataset_info(X)
            max_node = dataset_info["max_node"]

        self.input_dim_X = dataset_info["x_margins"].shape[0]
        self.input_dim_adj = dataset_info["e_margins"].shape[0]
        self.dataset_info = dataset_info
        self.max_node = max_node

        x_sde = load_sde(self.sde_type_x, self.sde_beta_min_x, self.sde_beta_max_x, self.sde_num_scales_x)
        adj_sde = load_sde(self.sde_type_adj, self.sde_beta_min_adj, self.sde_beta_max_adj, self.sde_num_scales_adj)

        self.loss_fn = get_sde_loss_fn(
            x_sde,
            adj_sde,
            train=True,
            reduce_mean=self.use_loss_reduce_mean,
            continuous=True,
            likelihood_weighting=False,
            eps=self.train_eps,
        )

        self.sampling_fn = get_pc_sampler(
            sde_x=x_sde,
            sde_adj=adj_sde,
            predictor=self.sampler_predictor,
            corrector=self.sampler_corrector,
            snr=self.sampler_snr,
            scale_eps=self.sampler_scale_eps,
            n_steps=self.sampler_n_steps,
            probability_flow=self.sampler_probability_flow,
            continuous=True,
            denoise=self.sampler_noise_removal,
            eps=self.sampler_eps,
            device=self.device,
        )


    def _initialize_model(
        self,
        model_class: Type[torch.nn.Module],
        checkpoint: Optional[Dict] = None
    ) -> torch.nn.Module:
        """Initialize the model with parameters or a checkpoint."""
        model_params = self._get_model_params(checkpoint)
        self.model = model_class(**model_params)
        self.model = self.model.to(self.device)
        
        if checkpoint is not None:
            self._setup_diffusion_params(checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        return self.model

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
    ) -> "GDSSMolecularGenerator":
        """Fit the model to the training data.

        Parameters
        ----------
        X_train : List[str]
            List of training data in SMILES format.

        Returns
        -------
        self : GDSSMolecularGenerator
            The fitted model.
        """
        X_train, _ = self._validate_inputs(X_train)
        self._setup_diffusion_params(X_train)
        self._initialize_model(self.model_class)
        self.model.initialize_parameters()

        optimizer, scheduler = self._setup_optimizers()
        train_dataset = self._convert_to_pytorch_data(X_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.fitting_loss = []
        self.fitting_epoch = 0
        for epoch in range(self.epochs):
            train_losses = self._train_epoch(train_loader, optimizer, epoch)
            self.fitting_loss.append(np.mean(train_losses).item())
            if scheduler:
                scheduler.step(np.mean(train_losses).item())

        self.fitting_epoch = epoch
        self.is_fitted_ = True
        return self
    
    def _train_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        losses = []
        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )
        active_index = self.dataset_info["active_index"]
        for step, batched_data in enumerate(iterator):
            batched_data = batched_data.to(self.device)
            optimizer.zero_grad()

            data_x = F.one_hot(batched_data.x, num_classes=118).float()[:, active_index]
            data_edge_attr = batched_data.edge_attr.float()
            X, E, node_mask = to_dense(data_x, batched_data.edge_index, data_edge_attr, batched_data.batch, self.max_node)
            
            loss_x, loss_adj = self.model.compute_loss(x=X, adj=E, flags=node_mask, loss_fn=self.loss_fn)
            loss = loss_x + loss_adj
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            optimizer.step()
            losses.append(loss.item())

            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Loss": f"{loss.item():.4f}", "Loss_x": f"{loss_x.item():.4f}", "Loss_adj": f"{loss_adj.item():.4f}"})
            
        return losses
    
    @torch.no_grad()
    def generate(self, num_nodes: Optional[Union[List[List], np.ndarray, torch.Tensor]] = None, batch_size: int = 32) -> List[str]:
        """Randomly generate molecules with specified node counts.

        Parameters
        ----------
        num_nodes : Optional[Union[List[List], np.ndarray, torch.Tensor]], default=None
            Number of nodes for each molecule in the batch. If None, samples from
            the training distribution. Can be provided as:
            - A list of lists
            - A numpy array of shape (batch_size, 1) 
            - A torch tensor of shape (batch_size, 1)
            
        batch_size : int, default=32
            Number of molecules to generate.

        Returns
        -------
        List[str]
            List of generated molecules in SMILES format.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before generating molecules.")

        if num_nodes is not None:
            batch_size = len(num_nodes)

        if num_nodes is None:
            num_nodes_dist = self.dataset_info["num_nodes_dist"]
            num_nodes = num_nodes_dist.sample_n(batch_size, self.device)
        elif isinstance(num_nodes, list):
            num_nodes = torch.tensor(num_nodes).to(self.device)
        elif isinstance(num_nodes, np.ndarray):
            num_nodes = torch.from_numpy(num_nodes).to(self.device)
        if num_nodes.dim() == 1:
            num_nodes = num_nodes.unsqueeze(-1)
        
        assert num_nodes.size(0) == batch_size
        arange = (
            torch.arange(self.max_node).to(self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        node_mask = arange < num_nodes

        if not hasattr(self, 'dataset_info') or self.dataset_info is None:
            raise ValueError("Dataset info not found. Please call setup_diffusion_params first.")
        
        shape_x = (
            batch_size,
            self.max_node,
            self.input_dim_X,
        )
        shape_adj = (batch_size, self.max_node, self.max_node)

        X, E, _ = self.sampling_fn(self.model.score_network_x, self.model.score_network_a, shape_x, shape_adj, node_mask)
        E = quantize_mol(E)
        X = X.argmax(dim=-1)

        molecule_list = []
        for i in range(batch_size):
            n = num_nodes[i][0].item()
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        smiles_list = graph_to_smiles(molecule_list, self.dataset_info["atom_decoder"])
        return smiles_list