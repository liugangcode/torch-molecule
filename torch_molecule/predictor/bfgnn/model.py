import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ...nn import GNN_node, GNN_node_Virtualnode, MLP
from ...utils import init_weights

class BFGNN(nn.Module):
    def __init__(
        self,
        num_task,
        num_layer,
        hidden_size=300,
        gnn_type="gin-virtual",
        drop_ratio=0.5,
        norm_layer="batch_norm",
        graph_pooling="max",
        augmented_feature=['maccs', 'morgan'],
        algorithm_aligned = 'bf'
    ):
        super(BFGNN, self).__init__()
        gnn_name = gnn_type.split("-")[0]
        self.algorithm_aligned = algorithm_aligned
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
                algorithm_aligned=algorithm_aligned
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
                algorithm_aligned=algorithm_aligned
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

    def compute_loss(self, batched_data, criterion, l1_penalty=1e-3):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        h_rep = self._augmented_graph_features(batched_data, h_rep)
        prediction = self.predictor(h_rep)
        target = batched_data.y.to(torch.float32)
        is_labeled = batched_data.y == batched_data.y
        loss = criterion(prediction.to(torch.float32)[is_labeled], target[is_labeled]).mean()
        l1_loss = sum(p.abs().sum() for p in self.graph_encoder.parameters()) + sum(p.abs().sum() for p in self.predictor.parameters())
        return loss + l1_penalty * l1_loss

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        h_rep = self._augmented_graph_features(batched_data, h_rep)
        prediction = self.predictor(h_rep)
        return {"prediction": prediction}