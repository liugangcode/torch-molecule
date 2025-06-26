import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ...nn import GNN_node, GNN_node_Virtualnode, MLP
from ...utils import init_weights
from ...utils.graph.features import allowable_features

import random

class_criterion = torch.nn.CrossEntropyLoss()

class GNN(nn.Module):
    def __init__(
        self,
        num_layer,
        hidden_size,
        drop_ratio=0.5,
        norm_layer="batch_norm",
        encoder_type="gin-virtual",
        readout="max",
        mask_num=0,
        mask_rate=0.15
    ):
        super(GNN, self).__init__()
        gnn_name = encoder_type.split("-")[0]
        decoding_size = len(allowable_features['possible_atomic_num_list'])
        self.hidden_size = hidden_size
        self.mask_num = mask_num
        self.mask_rate = mask_rate

        self.mask_atom_id = 119
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

        self.predictor = MLP(hidden_size, hidden_features=2 * hidden_size, out_features=decoding_size)
    
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

    def compute_loss(self, batched_data):
        masked_node_indices = list()

        # select indices of masked nodes
        for i in range(batched_data.batch[-1] + 1):
            idx = torch.nonzero((batched_data.batch == i).float()).squeeze(-1)
            num_node = idx.shape[0]
            if self.mask_num == 0:
                sample_size = int(num_node * self.mask_rate + 1)
            else:
                sample_size = min(self.mask_num, int(num_node * 0.5))
            masked_node_idx = random.sample(idx.tolist(), sample_size)
            masked_node_idx.sort()
            masked_node_indices += masked_node_idx

        batched_data.masked_node_indices = torch.tensor(masked_node_indices)

        batched_data.y = batched_data.x[masked_node_indices][:, 0]

        # mask nodes' features
        for node_idx in masked_node_indices:
            batched_data.x[node_idx] = torch.tensor([self.mask_atom_id - 1] + [0] * (batched_data.x.shape[1] - 1))
    
        # generate predictions
        h_node, _ = self.graph_encoder(batched_data)
        prediction_class = self.predictor(h_node[masked_node_indices])
        
        target_class = batched_data.y.to(torch.float32)      
        loss_class = class_criterion(prediction_class.to(torch.float32), target_class.long())

        return loss_class

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        return {"graph": h_rep, "node": h_node}