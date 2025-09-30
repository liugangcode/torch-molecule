import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .model import GraphTransformer
from .utils import PlaceHolder, to_dense, compute_dataset_info, DistributionNodes
from .flow_matching import NoiseDistribution, TimeDistorter, p_xt_g_x1, RateMatrixDesigner, sample_discrete_feature_noise, sample_discrete_features

from .extra_features import ExtraFeatures

from ...base import BaseMolecularGenerator
from ...utils import graph_from_smiles, graph_to_smiles

class DeFoGMolecularGenerator(BaseMolecularGenerator):
    """
    This generator implements the Discrete Flow Graph Model (DeFoG) for (multi-conditional and unconditional) molecular generation.
    It uses a flow matching approach with a graph transformer backbone.

    References
    ----------
    - DeFoG: A Discrete Flow Model for Graph Generation. ICML 2025.
      (https://openreview.net/forum?id=KPRIwWhqAZ)
    - Implementation based on: https://github.com/manuelmlmadeira/DeFoG

    Parameters
    ----------
    num_layer : int, default=6
        Number of transformer layers
    hidden_mlp_dims : Dict[str, int], default={'X': 256, 'E': 128, 'y': 128} if None
        Hidden dimensions for MLP layers in X (node dim), E (edge dim), and y (property dim) components
    hidden_dims : Dict[str, Any], default={'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128} if None
        Hidden dimensions for transformer components including attention heads and feed-forward layers
        Keys: 'dx' (node dim), 'de' (edge dim), 'dy' (property dim), 'n_head' (number of attention heads), 'dim_ffX' (feed-forward dim for node features), 'dim_ffE' (feed-forward dim for edge features), 'dim_ffy' (feed-forward dim for property features)
    transition : str, default='marginal'
        Transition type for flow matching.
        Options: 'marginal', 'absorbing', 'uniform', 'absorbfirst', 'argmax', 'edge_marginal', 'node_marginal'
    time_distortion : str, default="polydec"
        Time distortion schedule used during training/sampling.
        Options: 'identity', 'cos', 'revcos', 'polyinc', 'polydec'
    lambda_train : List[float], default=[5.0, 1.0] if None
        Loss weights: [edge_loss_weight, property_loss_weight]
    extra_features_type : str, default='rrwp'
        Extra feature type.
        Options: 'rrwp', 'rrwp_double', 'rrwp_only', 'rrwp_comp', 'cycles', 'eigenvalues', 'all'
    rrwp_steps : int, default=16
        Number of steps for (R)RWP features
    sample_steps : int, default=500
        Number of sampling steps during generation
    rdb : str, default="general"
        Rate matrix design (R^db) method.
        Options: 'general', 'marginal', 'column', 'entry'
    rdb_crit : str, default="p_x1_g_xt"
        Criterion used by 'column'/'entry' designs.
        Options: 'max_marginal', 'x_t', 'abs_state', 'p_x1_g_xt', 'x_1', 'p_xt_g_x1', 'xhat_t', 'first'
    eta : float, default=0
        Stochasticity scaling for R^db (higher = more stochastic)
    omega : float, default=0.1
        Target-guidance scaling for R^tg (higher = stronger guidance)
    guidance_weight : float, default=0.2
        Classifier-free guidance weight during sampling
    batch_size : int, default=128
        Batch size for training
    epochs : int, default=1000
        Number of training epochs
    learning_rate : float, default=0.0002
        Learning rate for optimization
    grad_clip_value : Optional[float], default=1.0
        Value for gradient clipping (None = no clipping)
    weight_decay : float, default=0.0
        Weight decay for optimization
    use_lr_scheduler : bool, default=False
        Whether to use learning rate scheduler
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate on plateau
    scheduler_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced
    task_type : List[str], default=[]
        List specifying type of each task ('regression' or 'classification')
    verbose : str, default="none"
        Whether to display progress info. Options are: "none", "progress_bar", "print_statement". If any other, "none" is automatically chosen.
    device : Optional[Union[torch.device, str]], default=None
        Device to run the model on (CPU or GPU)
    model_name : str, default="DeFoGMolecularGenerator"
        Name identifier for the model
    """

    def __init__(
        self,
        num_layer: int = 6,
        hidden_mlp_dims: Optional[Dict[str, int]] = None,
        hidden_dims: Optional[Dict[str, Any]] = None,
        transition: str = 'marginal',
        time_distortion: str = "polydec",
        lambda_train: Optional[List[float]] = None,
        extra_features_type: str = 'rrwp',
        rrwp_steps: int = 16,
        sample_steps: int = 500,
        rdb: str = "general",
        rdb_crit: str = "p_x1_g_xt",
        eta: float = 0,
        omega: float = 0.1,
        guidance_weight: float = 0.2,
        batch_size: int = 128,
        epochs: int = 1000,
        learning_rate: float = 0.0002,
        grad_clip_value: Optional[float] = 1.0,
        weight_decay: float = 0.0,
        use_lr_scheduler: bool = False,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        task_type: Optional[List[str]] = None,
        verbose: str = "none",
        *,
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "DeFoGMolecularGenerator",
    ):
        super().__init__(device=device, model_name=model_name, verbose=verbose)

        # Defaults for dict/list parameters
        if hidden_mlp_dims is None:
            hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}
        if hidden_dims is None:
            hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128}
        if lambda_train is None:
            lambda_train = [5.0, 1.0]

        # Model hyperparameters
        self.num_layer = num_layer
        self.hidden_mlp_dims = hidden_mlp_dims
        self.hidden_dims = hidden_dims

        # Flow Matching & Training parameters
        self.transition = transition
        self.time_distortion = time_distortion
        self.lambda_train = lambda_train
        self.extra_features_type = extra_features_type
        self.rrwp_steps = rrwp_steps

        # Sampling parameters
        self.sample_steps = sample_steps
        self.rdb = rdb
        self.rdb_crit = rdb_crit
        self.eta = eta
        self.omega = omega
        self.guidance_weight = guidance_weight

        # Standard training parameters (mirror GraphDiT style)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.grad_clip_value = grad_clip_value
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        # Task and verbosity
        if task_type is None:
            self.task_type = list()
        else:
            self.task_type = task_type

        # Book-keeping attributes
        self.fitting_loss = list()
        self.fitting_epoch = 0
        self.dataset_info = dict()
        self.model_class = GraphTransformer

        # Dimensions, to be set up later
        self.max_node = None
        self.input_dim_X = None
        self.input_dim_E = None
        self.input_dim_y = len(self.task_type)

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
            "max_node", "num_layer", "hidden_mlp_dims", "hidden_dims", 
            "input_dim_X", "input_dim_E", "input_dim_y", "task_type", 
            # Flow Matching parameters
            "transition", "time_distortion", "lambda_train", "extra_features_type", 
            "rrwp_steps", "sample_steps", "rdb", "rdb_crit", "eta", "omega", 
            "guidance_weight", "dataset_info",
            # Training Parameters
            "batch_size", "epochs", "learning_rate", "grad_clip_value", 
            "weight_decay", "use_lr_scheduler", "scheduler_factor", "scheduler_patience",
            # Other Parameters
            "fitting_epoch", "fitting_loss", "device", "verbose", "model_name"
        ]

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        """Get model parameters for initialization."""
        params = [
            "num_layer", "hidden_mlp_dims", "hidden_dims", "input_dim_X", 
            "input_dim_E", "input_dim_y", "task_type"
        ]
        
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            return {k: checkpoint["hyperparameters"][k] for k in params}
        
        return {k: getattr(self, k) for k in params}

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimization components including optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
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

    def _convert_to_pytorch_data(self, X, y=None):
        """Convert numpy arrays to PyTorch Geometric data format."""
        if self.verbose == "progress_bar":
            iterator = tqdm(enumerate(X), desc="Converting molecules to graphs", total=len(X))
        elif self.verbose == "print_statement":
            print("Converting molecules to graphs, preparing data for training...")
            iterator = enumerate(X)
        else:
            iterator = enumerate(X)

        pyg_graph_list = []
        for idx, smiles_or_mol in iterator:
            # For unconditional models, properties should be None
            properties = y[idx] if y is not None and len(self.task_type) > 0 else None
            graph = graph_from_smiles(smiles_or_mol, properties)
            g = Data()
            
            node_type = torch.from_numpy(graph['node_feat'][:, 0] - 1)
            if node_type.numel() <= 1:
                continue

            valid_mask = node_type >= 0
            if not valid_mask.all():
                # Get valid nodes and adjust edge indices
                valid_indices = torch.where(valid_mask)[0]
                index_map = -torch.ones(node_type.size(0), dtype=torch.long)
                index_map[valid_indices] = torch.arange(valid_indices.size(0))
                
                edge_index = torch.from_numpy(graph["edge_index"])
                valid_edges_mask = valid_mask[edge_index[0]] & valid_mask[edge_index[1]]
                valid_edge_index = edge_index[:, valid_edges_mask]
                remapped_edge_index = index_map[valid_edge_index]
                
                edge_attr = torch.from_numpy(graph["edge_feat"])[:, 0] + 1
                valid_edge_attr = edge_attr[valid_edges_mask]
                
                node_type = node_type[valid_mask]
                g.edge_index = remapped_edge_index
                g.edge_attr = valid_edge_attr.long()
            else:
                g.edge_index = torch.from_numpy(graph["edge_index"])
                edge_attr = torch.from_numpy(graph["edge_feat"])[:, 0] + 1
                g.edge_attr = edge_attr.long()
            
            node_type[node_type == 118] = 117 # Remap '*'
            g.x = node_type.long().squeeze(-1)
            
            # Only set g.y for conditional models
            if len(self.task_type) > 0:
                g.y = torch.from_numpy(graph["y"])
            
            pyg_graph_list.append(g)

        return pyg_graph_list

    def _setup_flow_params(self, X: Union[List, Dict]):
        """Sets up parameters and distributions required for the flow matching process."""
        if isinstance(X, dict):
            self.dataset_info = X["hyperparameters"]["dataset_info"]
        else:
            self.dataset_info = compute_dataset_info(X)

        class DeFoGDatasetInfos:
            def __init__(self, info, task_type):
                self.info = info
                self.output_dims = {'X': len(info['x_margins']), 'E': len(info['e_margins']), 'y': len(task_type)}
                self.input_dims = self.output_dims.copy()
                self.nodes_dist = info['num_nodes_dist']
                self.node_types = info['x_margins']
                self.edge_types = info['e_margins']
                self.atom_decoder = info['atom_decoder']

        defog_infos = DeFoGDatasetInfos(self.dataset_info, self.task_type)

        self.noise_dist = NoiseDistribution(self.transition, defog_infos)
        self.limit_dist = self.noise_dist.get_limit_dist()
        self.limit_dist.to_device(self.device)
        self.time_distorter = TimeDistorter(train_distortion=self.time_distortion, sample_distortion=self.time_distortion, alpha=1, beta=1)

        self.input_dim_X = self.noise_dist.x_num_classes
        self.input_dim_E = self.noise_dist.e_num_classes
        self.max_node = self.dataset_info['max_node']

    def _initialize_model(
        self,
        model_class: Type[torch.nn.Module],
        checkpoint: Optional[Dict] = None
    ) -> torch.nn.Module:
        """Initialize the model with parameters or a checkpoint."""
        
        if checkpoint is not None:
            # When loading from checkpoint, we need to set up flow params first
            self._setup_flow_params(checkpoint)
        
        # Initialize ExtraFeatures
        class DeFoGDatasetInfosForFeatures:
            def __init__(self, max_n_nodes):
                self.max_n_nodes = max_n_nodes
        
        feature_infos = DeFoGDatasetInfosForFeatures(self.max_node)
        self.extra_features_computer = ExtraFeatures(
            self.extra_features_type, self.rrwp_steps, feature_infos
        )

        # Temp calculation to get feature dimensions
        # This is a bit of a hack, but necessary to get the dims before full model init
        dummy_x = torch.zeros(1, self.max_node, self.input_dim_X, device=self.device)
        dummy_e = torch.zeros(1, self.max_node, self.max_node, self.input_dim_E, device=self.device)
        dummy_y = torch.zeros(1, self.input_dim_y, device=self.device)
        dummy_mask = torch.ones(1, self.max_node, device=self.device).bool()
        dummy_t = torch.zeros(1, 1, device=self.device)
        
        dummy_noisy_data = {'X_t': dummy_x, 'E_t': dummy_e, 'y_t': dummy_y, 't': dummy_t, 'node_mask': dummy_mask}
        extra_data = self._compute_extra_data(dummy_noisy_data)
        
        self.input_dim_X_extra = extra_data.X.shape[-1]
        self.input_dim_E_extra = extra_data.E.shape[-1]
        self.input_dim_y_extra = extra_data.y.shape[-1]

        model_params = {
            'n_layers': self.num_layer,
            'input_dims': {
                'X': self.input_dim_X + self.input_dim_X_extra,
                'E': self.input_dim_E + self.input_dim_E_extra,
                'y': self.input_dim_y + self.input_dim_y_extra,
            },
            'hidden_mlp_dims': self.hidden_mlp_dims,
            'hidden_dims': self.hidden_dims,
            'output_dims': {'X': self.input_dim_X, 'E': self.input_dim_E, 'y': self.input_dim_y},
            'act_fn_in': nn.ReLU(),
            'act_fn_out': nn.ReLU(),
        }
        self.model = model_class(**model_params)
        self.model = self.model.to(self.device)
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=self.rdb,
            rdb_crit=self.rdb_crit,
            eta=self.eta,
            omega=self.omega,
            limit_dist=self.limit_dist
        )

    def fit(self, X_train: List[str], y_train: Optional[Union[List, np.ndarray]] = None) -> "DeFoGMolecularGenerator":
        num_task = len(self.task_type)
        
        # For unconditional models, ensure y_train is None
        if num_task == 0 and y_train is not None:
            print("Warning: y_train provided for unconditional model. Ignoring y_train.")
            y_train = None
        
        X_train, y_train = self._validate_inputs(X_train, y_train, num_task=num_task)
        self._setup_flow_params(X_train)
        self._initialize_model(self.model_class)

        optimizer, scheduler = self._setup_optimizers()
        train_dataset = self._convert_to_pytorch_data(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Calculate total steps for global progress bar
        total_steps = self.epochs * len(train_loader)
        global_progress = None
        if self.verbose == "progress_bar":
            global_progress = tqdm(total=total_steps, desc="Training Progress", leave=True)
        self.fitting_loss = []
        for epoch in range(self.epochs):
            train_losses = self._train_epoch(train_loader, optimizer, epoch, global_progress)
            avg_loss = np.mean(train_losses)
            self.fitting_loss.append(avg_loss)
            if scheduler:
                scheduler.step(avg_loss)
        
        if global_progress:
            global_progress.close()
        
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, epoch, global_progress=None):
        self.model.train()
        losses = []

        active_index = self.dataset_info["active_index"]
        for batched_data in train_loader:
            batched_data = batched_data.to(self.device)
            optimizer.zero_grad()
            
            data_x = F.one_hot(batched_data.x, num_classes=118).float()[:, active_index]
            data_edge_attr = F.one_hot(batched_data.edge_attr, num_classes=5).float()
            dense_data, node_mask = to_dense(data_x, batched_data.edge_index, data_edge_attr, batched_data.batch)
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            
            # Process y data
            if hasattr(batched_data, "y") and len(self.task_type) > 0:  # Conditional model
                y = batched_data.y
                if torch.rand(1) < 0.1:  # 10% probability to use unconditional training
                    y = torch.ones_like(y, device=self.device) * -1
            else:  # Unconditional model
                # Ensure y has correct dimensions
                batch_size = X.size(0)
                y = torch.zeros(batch_size, self.input_dim_y, device=self.device)
            
            noisy_data = self.apply_noise(X, E, y, node_mask)
            extra_data = self._compute_extra_data(noisy_data)
            
            pred = self.model(
                torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float(),
                torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float(),
                torch.cat((noisy_data["y_t"], extra_data.y), dim=1).float(),
                node_mask
            )
            
            loss_X = F.cross_entropy(pred.X.transpose(1, 2), X.argmax(-1), reduction='none')
            loss_E = F.cross_entropy(pred.E.transpose(1, 3), E.argmax(-1), reduction='none')
            
            masked_loss_X = (loss_X * node_mask).sum() / node_mask.sum()
            e_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            masked_loss_E = (loss_E * e_mask).sum() / e_mask.sum()
            
            if len(self.task_type) > 0 and y.shape[-1] > 0:
                # For conditional models, compute loss on y predictions
                # Use MSE loss for regression tasks, similar to original DeFoG
                loss_y = F.mse_loss(pred.y, y, reduction='mean')
            else:
                loss_y = torch.tensor(0.0, device=self.device)
            
            loss = masked_loss_X + self.lambda_train[0] * masked_loss_E + self.lambda_train[1] * loss_y

            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()
            
            losses.append(loss.item())
            
            # Update global progress bar
            if global_progress:
                global_progress.set_postfix({
                    "Epoch": f"{epoch+1}",
                    "Loss": f"{loss.item():.4f}",
                    "Loss_X": f"{masked_loss_X.item():.4f}",
                    "Loss_E": f"{masked_loss_E.item():.4f}",
                    "Loss_y": f"{loss_y.item():.4f}"
                })
                global_progress.update(1)

        return losses

    def apply_noise(self, X, E, y, node_mask):
        bs = X.size(0)
        t_float = self.time_distorter.train_ft(bs, self.device)
        
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        
        prob_X_t, prob_E_t = p_xt_g_x1(X_1_label, E_1_label, t_float, self.limit_dist)
        
        sampled_t = sample_discrete_features(prob_X_t, prob_E_t, node_mask=node_mask)
        
        X_t = F.one_hot(sampled_t.X, num_classes=self.input_dim_X)
        E_t = F.one_hot(sampled_t.E, num_classes=self.input_dim_E)
        
        z_t = PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
        
        return {"t": t_float, "X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask}
    
    def _compute_extra_data(self, noisy_data):
        """Computes extra features for the model, like time embedding and RRWP."""
        extra_features = self.extra_features_computer(noisy_data)
        
        t = noisy_data['t']
        extra_y = torch.cat((extra_features.y, t), dim=1)
        
        return PlaceHolder(X=extra_features.X, E=extra_features.E, y=extra_y)

    @torch.no_grad()
    def generate(self, labels: Optional[Union[List[List], np.ndarray, torch.Tensor]] = None, num_nodes: Optional[Union[List[List], np.ndarray, torch.Tensor]] = None, batch_size: int = 32) -> List[str]:
        """Generate molecules with specified properties and optional node counts.

        Parameters
        ----------
        labels : Optional[Union[List[List], np.ndarray, torch.Tensor]], default=None
            Target properties for the generated molecules. Can be provided as:
            - A list of lists for multiple properties 
            - A numpy array of shape (batch_size, n_properties)
            - A torch tensor of shape (batch_size, n_properties)
            For single label (properties values), can also be provided as 1D array/tensor.
            If None, generates unconditional samples.
            
        num_nodes : Optional[Union[List[List], np.ndarray, torch.Tensor]], default=None
            Number of nodes for each molecule in the batch. If None, samples from
            the training distribution. Can be provided as:
            - A list of lists
            - A numpy array of shape (batch_size, 1) 
            - A torch tensor of shape (batch_size, 1)
            
        batch_size : int, default=32
            Number of molecules to generate. Only used if labels is None.

        Returns
        -------
        List[str]
            List of generated molecules in SMILES format.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before generating molecules.")
        if self.input_dim_X is None or self.input_dim_E is None or self.max_node is None:
            raise ValueError(f"Model may not be fitted correctly as one of below attributes is not set: input_dim_X={self.input_dim_X}, input_dim_E={self.input_dim_E}, max_node={self.max_node}")
        
        if len(self.task_type) > 0 and labels is None:
            raise ValueError(f"labels must be provided if task_type is not empty: {self.task_type}")

        if labels is not None and num_nodes is not None:
            assert len(labels) == len(num_nodes), "labels and num_nodes must have the same batch size"
        
        if labels is not None:
            if num_nodes is not None:
                assert len(labels) == len(num_nodes), "labels and num_nodes must have the same batch size"
            batch_size = len(labels)
        elif num_nodes is not None:
            batch_size = len(num_nodes)

        # Convert properties to 2D tensor if needed
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        elif isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if labels is not None and labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        
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

        self.model.eval()
        
        z_T = sample_discrete_feature_noise(self.limit_dist, node_mask)
        X, E = z_T.X, z_T.E
        
        if len(self.task_type) > 0 and labels is not None:
            y = labels.to(self.device).float()
        else:
            # unconditional
            y = torch.zeros(batch_size, self.input_dim_y, device=self.device)

        for t_int in tqdm(reversed(range(0, self.sample_steps)), desc="Generating", total=self.sample_steps, disable=(not self.verbose=="progress_bar")):
            t_array = t_int * torch.ones((batch_size, 1), device=self.device)
            s_array = (t_int + 1) * torch.ones((batch_size, 1), device=self.device)
            
            t_norm = t_array / self.sample_steps
            s_norm = s_array / self.sample_steps

            t_norm_distorted = self.time_distorter.sample_ft(t_norm, self.time_distortion)
            s_norm_distorted = self.time_distorter.sample_ft(s_norm, self.time_distortion)

            sampled_s, _ = self._sample_step(t_norm_distorted, s_norm_distorted, X, E, y, node_mask)
            X, E = sampled_s.X, sampled_s.E

        sampled_s = PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)
        X_final, E_final, _ = self.noise_dist.ignore_virtual_classes(sampled_s.X, sampled_s.E, sampled_s.y)
        
        molecule_list = []
        for i in range(batch_size):
            n = num_nodes[i].item()
            atom_types = X_final[i, :n].cpu()
            edge_types = E_final[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            
        return graph_to_smiles(molecule_list, self.dataset_info["atom_decoder"])

    def _sample_step(self, t, s, X_t, E_t, y_t, node_mask):
        dt = (s - t)[0]
        
        def get_rates(y_in):
            noisy_data = {"X_t": X_t, "E_t": E_t, "y_t": y_in, "t": t, "node_mask": node_mask}
            extra_data = self._compute_extra_data(noisy_data)
            
            pred = self.model(
                torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float(),
                torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float(),
                torch.cat((noisy_data["y_t"], extra_data.y), dim=1).float(),
                node_mask
            )
            
            pred_X = F.softmax(pred.X, dim=-1)
            pred_E = F.softmax(pred.E, dim=-1)
            return self.rate_matrix_designer.compute_graph_rate_matrix(t, node_mask, (X_t, E_t), (pred_X, pred_E))

        R_t_X, R_t_E = get_rates(y_t)
        
        is_conditional = len(self.task_type) > 0 and y_t.shape[-1] > 0 and y_t.sum() != 0
        if is_conditional and self.guidance_weight > 0:
            # Use -1 as unconditional signal, similar to original DeFoG
            uncond_y = torch.ones_like(y_t, device=self.device) * -1
            R_t_X_uncond, R_t_E_uncond = get_rates(uncond_y)
            
            # Use logarithmic interpolation
            R_t_X = torch.exp(
                torch.log(R_t_X_uncond + 1e-6) * (1 - self.guidance_weight) 
                + torch.log(R_t_X + 1e-6) * self.guidance_weight
            )
            R_t_E = torch.exp(
                torch.log(R_t_E_uncond + 1e-6) * (1 - self.guidance_weight) 
                + torch.log(R_t_E + 1e-6) * self.guidance_weight
            )

        step_probs_X, step_probs_E = self._compute_step_probs(R_t_X, R_t_E, X_t, E_t, dt)
        
        if s[0] >= 1.0:
            noisy_data = {"X_t": X_t, "E_t": E_t, "y_t": y_t, "t": t, "node_mask": node_mask}
            extra_data = self._compute_extra_data(noisy_data)
            pred = self.model(
                torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float(),
                torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float(),
                torch.cat((noisy_data["y_t"], extra_data.y), dim=1).float(),
                node_mask
            )
            step_probs_X, step_probs_E = F.softmax(pred.X, dim=-1), F.softmax(pred.E, dim=-1)

        sampled_s = sample_discrete_features(step_probs_X, step_probs_E, node_mask=node_mask)
        
        X_s = F.one_hot(sampled_s.X, num_classes=self.input_dim_X).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.input_dim_E).float()

        if is_conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = PlaceHolder(X=X_s, E=E_s, y=y_to_save).mask(node_mask).type_as(y_t)
        out_discrete = PlaceHolder(X=X_s, E=E_s, y=y_to_save).mask(node_mask, collapse=True).type_as(y_t)
        
        return out_one_hot, out_discrete

    def _compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt):
        step_probs_X = (R_t_X * dt).clamp(0, 1)
        step_probs_E = (R_t_E * dt).clamp(0, 1)
        
        step_probs_X.scatter_(-1, X_t.argmax(-1, keepdim=True), 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1, keepdim=True), 0.0)
        
        diag_X = (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        diag_E = (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        
        step_probs_X.scatter_(-1, X_t.argmax(-1, keepdim=True), diag_X)
        step_probs_E.scatter_(-1, E_t.argmax(-1, keepdim=True), diag_E)
        
        return step_probs_X, step_probs_E
