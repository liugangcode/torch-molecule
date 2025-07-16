import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ...nn import GNN_node, GNN_node_Virtualnode, MLP
from ...utils import init_weights

class GNN(nn.Module):
    def __init__(
        self,
        num_task,
        num_layer,
        hidden_size=300,
        gnn_type="gin-virtual",
        drop_ratio=0.5,
        norm_layer="batch_norm",
        graph_pooling="max",
        augmented_feature=['maccs', 'morgan']
    ):
        super(GNN, self).__init__()
        gnn_name = gnn_type.split("-")[0]
        self.num_task = num_task
        self.hidden_size = hidden_size

        if "virtual" in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                hidden_size,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        else:
            self.graph_encoder = GNN_node(
                num_layer,
                hidden_size,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Invalid graph pooling type {graph_pooling}.")

        graph_dim = hidden_size
        self.augmented_feature = augmented_feature
        if augmented_feature:
            if "morgan" in augmented_feature:
                graph_dim += 1024
            if "maccs" in augmented_feature:
                graph_dim += 167
        self.predictor = MLP(graph_dim, hidden_features=2 * hidden_size, out_features=num_task)
    
    def initialize_parameters(self, seed=None):
        """
        Randomly initialize all model parameters using the init_weights function.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize the main components
        init_weights(self.graph_encoder)
        init_weights(self.predictor)
        
        # Reset all parameters using PyTorch Geometric's reset function
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                init_weights(module)
        
        self.apply(reset_parameters)

    def _augmented_graph_features(self, batched_data, h_rep):
        if self.augmented_feature:
            if 'morgan' in self.augmented_feature:
                morgan = batched_data.morgan.type_as(h_rep)
                h_rep = torch.cat((h_rep, morgan), dim=1)
            if 'maccs' in self.augmented_feature:
                maccs = batched_data.maccs.type_as(h_rep)
                h_rep = torch.cat((h_rep, maccs), dim=1)
        return h_rep

    def compute_loss(self, batched_data, criterion, scale=1.0, penalty_weight=1.0):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        h_rep = self._augmented_graph_features(batched_data, h_rep)
        prediction = self.predictor(h_rep)
        target = batched_data.y.to(torch.float32)
        is_labeled = batched_data.y == batched_data.y

        dummy = torch.nn.Parameter(torch.Tensor([scale])).to(prediction.device)
        losses_erm = criterion(prediction.to(torch.float32)[is_labeled] * dummy, target[is_labeled])
        
        environments = batched_data.environment
        if environments.dim() > 1 and environments.shape[1] == 1:
            environments = environments.expand(-1, is_labeled.shape[1])
        environments = environments[is_labeled]
        unique_envs = environments.unique()  
        env_losses = []
        for env in unique_envs:
            env_mask = environments == env
            env_loss = losses_erm[env_mask].mean()
            env_grad = torch.autograd.grad(env_loss, dummy, create_graph=True)[0]
            env_losses.append(torch.sum(env_grad**2))
        
        penalty = torch.sum(torch.stack(env_losses))
        total_loss = losses_erm.mean() + penalty_weight * penalty
        if penalty_weight > 1.0:
            total_loss /= penalty_weight
        return total_loss, losses_erm.mean(), penalty

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        h_rep = self._augmented_graph_features(batched_data, h_rep)
        prediction = self.predictor(h_rep)
        return {"prediction": prediction}