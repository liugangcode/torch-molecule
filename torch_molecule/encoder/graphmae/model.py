from functools import partial

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch.nn.functional as F

from ...nn import GNN_node, GNN_node_Virtualnode
from ...nn.gnn import GINConv, GCNConv
from ...utils import init_weights

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

class GNN(nn.Module):
    def __init__(
        self,
        num_layer,
        hidden_size,
        drop_ratio=0.5,
        norm_layer="batch_norm",
        encoder_type="gin-virtual",
        readout="sum",
        mask_edge=False,
        predictor_type="gin",
    ):
        super(GNN, self).__init__()
        gnn_name = encoder_type.split("-")[0]
        self.num_atom_type = 119
        self.num_edge_type = 5
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

        self.atom_predictor = Predictor(hidden_size, self.num_atom_type, nn_type=predictor_type)
        self.mask_edge = mask_edge
        if mask_edge:
            self.bond_predictor = Predictor(hidden_size, self.num_edge_type, nn_type="linear")
    
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
        init_weights(self.atom_predictor)
        if self.mask_edge:
            init_weights(self.bond_predictor)
        
        # Reset all parameters using PyTorch Geometric's reset function
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                init_weights(module)
        
        self.apply(reset_parameters)

    def compute_loss(self, batched_data, alpha_l=1):
        criterion = partial(sce_loss, alpha=alpha_l)

        h_node, _ = self.graph_encoder(batched_data)

        node_attr_label = batched_data.node_attr_label
        masked_node_indices = batched_data.masked_atom_indices
        pred_node = self.atom_predictor(h_node, batched_data.edge_index, batched_data.edge_attr, masked_node_indices)
        loss_atom = criterion(pred_node[masked_node_indices], node_attr_label.to(pred_node.dtype))
        if self.mask_edge:
            masked_edge_index = batched_data.edge_index[:, batched_data.connected_edge_indices]
            edge_rep = h_node[masked_edge_index[0]] + h_node[masked_edge_index[1]]
            pred_edge = self.bond_predictor(edge_rep)
            loss_edge = criterion(pred_edge, batched_data.edge_attr_label.to(pred_edge.dtype))
        else:
            loss_edge = torch.tensor(0.0)
        return loss_atom, loss_edge

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        return {"graph": h_rep, "node": h_node}

class Predictor(torch.nn.Module):
    def __init__(self, hidden_size, output_size, nn_type = "gin"):
        super().__init__()
        self._dec_type = nn_type 
        if "gin" in nn_type:
            self.conv = GINConv(hidden_size, output_size)
        elif "gcn" in nn_type:
            self.conv = GCNConv(hidden_size, output_size)
        elif "linear" in nn_type:
            self.dec = torch.nn.Linear(hidden_size, output_size)
        else:
            raise NotImplementedError(f"{nn_type}")
        if "linear" not in nn_type: 
            self.activation = torch.nn.PReLU() 
            self.enc_to_dec = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, edge_index=None, edge_attr=None, mask_node_indices=None):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            assert edge_index is not None and edge_attr is not None and mask_node_indices is not None
            x = self.activation(x)
            x = self.enc_to_dec(x)
            x[mask_node_indices] = 0
            out = self.conv(x, edge_index, edge_attr)
        return out