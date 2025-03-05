import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ...nn import GNN_node, GNN_node_Virtualnode
from ...utils import init_weights

criterion = torch.nn.BCEWithLogitsLoss()

class GNN(nn.Module):
    def __init__(
        self,
        num_layer,
        hidden_size,
        drop_ratio=0.5,
        norm_layer="batch_norm",
        encoder_type="gin-virtual",
        readout="max"
    ):
        super(GNN, self).__init__()
        gnn_name = encoder_type.split("-")[0]
        self.hidden_size = hidden_size
        encoder_params = {
            "num_layer": num_layer,
            "hidden_size": hidden_size,
            "JK": "last", 
            "drop_ratio": drop_ratio,
            "residual": True,
            "gnn_name": gnn_name,
            "norm_layer": norm_layer
        }
        
        # Choose encoder type based on encoder_model
        encoder_class = GNN_node_Virtualnode if "virtual" in encoder_type else GNN_node
        self.graph_encoder = encoder_class(**encoder_params)
        pooling_funcs = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool
        }
        self.pool = pooling_funcs.get(readout)
        if self.pool is None:
            raise ValueError(f"Invalid graph pooling type {readout}.")

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
        
        # Reset all parameters using PyTorch Geometric's reset function
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                init_weights(module)
        
        self.apply(reset_parameters)

    def compute_loss(self, batched_data):
        # sample negative edges
        num_nodes = batched_data.num_nodes
        num_edges = batched_data.num_edges

        edge_set = set([str(batched_data.edge_index[0, i].cpu().item()) + "," + str(
            batched_data.edge_index[1, i].cpu().item()) for i in range(batched_data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5 * num_edges):
            node1 = redandunt_sample[0, i].cpu().item()
            node2 = redandunt_sample[1, i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges / 2:
                break

        batched_data.negative_edge_index = redandunt_sample[:, sampled_ind]
    
        # generate predictions
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        
        positive_score = torch.sum(h_node[batched_data.edge_index[0, ::2]] * h_node[batched_data.edge_index[1, ::2]], dim = 1)
        negative_score = torch.sum(h_node[batched_data.negative_edge_index[0]] * h_node[batched_data.negative_edge_index[1]], dim = 1)
        
        loss_class = criterion(positive_score, torch.ones_like(positive_score)) + criterion(negative_score, torch.zeros_like(negative_score))
        
        return loss_class

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        return {"graph": h_rep, "node": h_node}