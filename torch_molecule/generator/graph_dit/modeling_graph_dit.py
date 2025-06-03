import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Type
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .transformer import Transformer
from .utils import PlaceHolder, to_dense, compute_dataset_info
from .diffusion import NoiseScheduleDiscrete, MarginalTransition, sample_discrete_features, sample_discrete_feature_noise, reverse_diffusion

from ...base import BaseMolecularGenerator
from ...utils import graph_from_smiles, graph_to_smiles

@dataclass
class GraphDITMolecularGenerator(BaseMolecularGenerator):
    """
    This generator implements the graph diffusion transformer for (multi-conditional and unconditional) molecular generation.

    References
    ----------
    - Graph Diffusion Transformers for Multi-Conditional Molecular Generation. NeurIPS 2024.
      https://openreview.net/forum?id=cfrDLD1wfO
    - Implementation: https://github.com/liugangcode/Graph-DiT

    Parameters
    ----------
    num_layer : int, default=6
        Number of transformer layers
    hidden_size : int, default=1152
        Dimension of hidden layers
    dropout : float, default=0.0
        Dropout rate for transformer layers
    drop_condition : float, default=0.0
        Dropout rate for condition embedding
    num_head : int, default=16
        Number of attention heads in transformer
    mlp_ratio : float, default=4
        Ratio of MLP hidden dimension to transformer hidden dimension
    task_type : List[str], default=[]
        List specifying type of each task ('regression' or 'classification')
    timesteps : int, default=500
        Number of diffusion timesteps
    batch_size : int, default=128
        Batch size for training
    epochs : int, default=10000
        Number of training epochs
    learning_rate : float, default=0.0002
        Learning rate for optimization
    grad_clip_value : Optional[float], default=None
        Value for gradient clipping (None = no clipping)
    weight_decay : float, default=0.0
        Weight decay for optimization
    lw_X : float, default=1
        Loss weight for node reconstruction
    lw_E : float, default=5
        Loss weight for edge reconstruction
    guide_scale : float, default=2.0
        Scale factor for classifier-free guidance during sampling
    use_lr_scheduler : bool, default=False
        Whether to use learning rate scheduler
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate on plateau
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced
    verbose : bool, default=False
        Whether to display progress bars and logs
    """
    
    # Model parameters
    num_layer: int = 6
    hidden_size: int = 1152
    dropout: float = 0.
    drop_condition: float = 0.
    num_head: int = 16
    mlp_ratio: float = 4
    task_type: List[str] = field(default_factory=list)

    # Diffusion parameters
    timesteps: int = 500
    
    # Training parameters
    batch_size: int = 128
    epochs: int = 10000
    learning_rate: float = 0.0002
    grad_clip_value: Optional[float] = None
    weight_decay: float = 0.0
    lw_X: float = 1
    lw_E: float = 5
    # Sampling parameters
    guide_scale: float = 2.
    # Scheduler parameters
    use_lr_scheduler: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    verbose: bool = False

    # Attributes
    model_name: str = "GraphDITMolecularGenerator"
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    dataset_info: Dict[str, Any] = field(default_factory=dict, init=False)
    model_class: Type[Transformer] = field(default=Transformer, init=False)

    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__post_init__()
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
            "max_node", "hidden_size", "num_layer", "num_head",
            "mlp_ratio", "dropout", "drop_condition", "input_dim_X", "input_dim_E", "input_dim_y",
            "task_type",
            # Diffusion parameters  
            "timesteps", "dataset_info",
            # Training Parameters
            "batch_size", "epochs", "learning_rate", "grad_clip_value", 
            "weight_decay", "lw_X", "lw_E",
            # Scheduler Parameters
            "use_lr_scheduler", "scheduler_factor", "scheduler_patience",
            # Sampling Parameters
            "guide_scale",
            # Other Parameters
            "fitting_epoch", "fitting_loss", "device", "verbose", "model_name"
        ]
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = ["max_node", "hidden_size", "num_layer", "num_head", "mlp_ratio", 
                 "dropout", "drop_condition", "input_dim_X", "input_dim_E", "input_dim_y", "task_type"]
        
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
            timesteps = X["hyperparameters"]["timesteps"] 
            max_node = X["hyperparameters"]["max_node"]
        else:
            assert isinstance(X, list)
            dataset_info = compute_dataset_info(X)
            timesteps = self.timesteps
            max_node = dataset_info["max_node"]

        self.input_dim_X = dataset_info["x_margins"].shape[0]
        self.input_dim_E = dataset_info["e_margins"].shape[0]
        self.dataset_info = dataset_info
        self.timesteps = timesteps
        self.max_node = max_node

        x_limit = dataset_info["x_margins"].to(self.device)
        e_limit = dataset_info["e_margins"].to(self.device)
        xe_conditions = dataset_info["xe_conditions"].to(self.device)
        ex_conditions = dataset_info["ex_conditions"].to(self.device)

        self.transition_model = MarginalTransition(x_limit, e_limit, xe_conditions, ex_conditions, self.max_node)
        self.limit_dist = PlaceHolder(X=x_limit, E=e_limit, y=None)
        self.noise_schedule = NoiseScheduleDiscrete(timesteps=self.timesteps).to(self.device)

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
        y_train: Optional[Union[List, np.ndarray]] = None,
    ) -> "GraphDITMolecularGenerator":
        num_task = len(self.task_type)
        X_train, y_train = self._validate_inputs(X_train, y_train, num_task=num_task)
        self._setup_diffusion_params(X_train)
        self._initialize_model(self.model_class)
        self.model.initialize_parameters()

        optimizer, scheduler = self._setup_optimizers()
        train_dataset = self._convert_to_pytorch_data(X_train, y_train)
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
            data_edge_attr = F.one_hot(batched_data.edge_attr, num_classes=5).float()
            dense_data, node_mask = to_dense(data_x, batched_data.edge_index, data_edge_attr, batched_data.batch, self.max_node)
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            noisy_data = self.apply_noise(X, E, batched_data.y, node_mask)

            loss, loss_X, loss_E = self.model.compute_loss(noisy_data, true_X=X, true_E=E, lw_X=self.lw_X, lw_E=self.lw_E)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            optimizer.step()
            losses.append(loss.item())
            
            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Loss": f"{loss.item():.4f}", "Loss_X": f"{loss_X.item():.4f}", "Loss_E": f"{loss_E.item():.4f}"})
            
        return losses

    def apply_noise(self, X, E, y, node_mask) -> Dict[str, Any]:
        t_int = torch.randint(0, self.timesteps + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.timesteps
        s_float = s_int / self.timesteps

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        
        bs, n, _ = X.shape
        X_all = torch.cat([X, E.reshape(bs, n, -1)], dim=-1)
        prob_all = X_all @ Qtb.X
        probX = prob_all[:, :, :self.input_dim_X]
        probE = prob_all[:, :, self.input_dim_X:].reshape(bs, n, n, -1)

        # check whether X_all/prob_all/probX/probE contain nan
        
        sampled_t = sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.input_dim_X)
        E_t = F.one_hot(sampled_t.E, num_classes=self.input_dim_E)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t': t_float * self.timesteps, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        
        return noisy_data

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

        if not hasattr(self, 'limit_dist') or self.limit_dist is None:
            raise ValueError("Limit distribution not found. Please call setup_diffusion_params first.")
        if not hasattr(self, 'dataset_info') or self.dataset_info is None:
            raise ValueError("Dataset info not found. Please call setup_diffusion_params first.")
        
        z_T = sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E = z_T.X, z_T.E

        assert (E == torch.transpose(E, 1, 2)).all()

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        if labels is not None:
            y = labels.to(self.device).float()
        else:
            y = None

        self.model.eval()
        for s_int in reversed(range(0, self.timesteps)):
            s_array = s_int * torch.ones((batch_size, 1)).float().to(self.device)
            t_array = s_array + 1
            s_norm = s_array / self.timesteps
            t_norm = t_array / self.timesteps

            # Sample z_s
            sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        molecule_list = []
        for i in range(batch_size):
            n = num_nodes[i][0].item()
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        smiles_list = graph_to_smiles(molecule_list, self.dataset_info["atom_decoder"])
        return smiles_list

    def sample_p_zs_given_zt(
        self, s, t, X_t, E_t, properties, node_mask
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, _ = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": properties,
            "t": t,
            "node_mask": node_mask,
        }

        def get_prob(noisy_data, unconditioned=False):
            pred = self.model(noisy_data, unconditioned=unconditioned)

            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            device = pred_X.device
            # Retrieve transitions matrix
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device)
            Qt = self.transition_model.get_Qt(beta_t, device)

            Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
            predX_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)

            # raise ValueError('stop here')
            unnormalized_probX_all = reverse_diffusion(
                predX_0=predX_all, X_t=Xt_all, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )

            unnormalized_prob_X = unnormalized_probX_all[:, :, : self.input_dim_X]
            unnormalized_prob_E = unnormalized_probX_all[
                :, :, self.input_dim_X :
            ].reshape(bs, n * n, -1)

            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5

            prob_X = unnormalized_prob_X / torch.sum(
                unnormalized_prob_X, dim=-1, keepdim=True
            )  # bs, n, d_t-1
            prob_E = unnormalized_prob_E / torch.sum(
                unnormalized_prob_E, dim=-1, keepdim=True
            )  # bs, n, d_t-1
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

            return prob_X, prob_E

        prob_X, prob_E = get_prob(noisy_data)

        ### Guidance
        if self.guide_scale is not None and self.guide_scale != 1:
            uncon_prob_X, uncon_prob_E = get_prob(noisy_data, unconditioned=True)
            prob_X = (
                uncon_prob_X
                * (prob_X / uncon_prob_X.clamp_min(1e-5)) ** self.guide_scale
            )
            prob_E = (
                uncon_prob_E
                * (prob_E / uncon_prob_E.clamp_min(1e-5)) ** self.guide_scale
            )
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-5)

        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.input_dim_X).to(self.device).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.input_dim_E).to(self.device).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = PlaceHolder(X=X_s, E=E_s, y=properties)

        return out_one_hot.mask(node_mask)