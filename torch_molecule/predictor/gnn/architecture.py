import os
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ..components.gnn_components import GNN_node, GNN_node_Virtualnode
from ...utils import init_weights

class GNN(nn.Module):
    def __init__(
        self,
        num_tasks,
        num_layer,
        emb_dim=300,
        gnn_type="gin-virtual",
        drop_ratio=0.5,
        norm_layer="batch_norm",
        graph_pooling="max"
    ):
        super(GNN, self).__init__()
        gnn_name = gnn_type.split("-")[0]
        self.num_tasks = num_tasks
        self.emb_dim = emb_dim

        if "virtual" in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        else:
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
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
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(2 * emb_dim, self.num_tasks),
        )
    
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

    def compute_loss(self, batched_data, criterion):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        prediction = self.predictor(h_rep)
        target = batched_data.y.to(torch.float32)
        is_labeled = batched_data.y == batched_data.y
        loss = criterion(prediction.to(torch.float32)[is_labeled], target[is_labeled])
        return loss

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        prediction = self.predictor(h_rep)
        return {"prediction": prediction}