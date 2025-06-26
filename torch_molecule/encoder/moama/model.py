import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ...nn import GNN_node, GNN_node_Virtualnode, GCNConv, GINConv
from ...utils import init_weights

from .utils import get_mask_indices, get_fingerprint_loss
from ...utils.graph.features import allowable_features

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
        mask_rate=0.15,
        lw_rec=0.5
    ):
        super(GNN, self).__init__()
        gnn_name = encoder_type.split("-")[0]
        decoding_size = len(allowable_features['possible_atomic_num_list'])
        
        self.mask_atom_id = 119
        self.hidden_size = hidden_size
        self.mask_rate = mask_rate
        self.lw_rec = lw_rec

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

        self.predictor = GNN_Decoder(hidden_size, decoding_size)
    
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
        masked_node_indices = get_mask_indices(batched_data, self.mask_rate)
        batched_data.masked_node_indices = torch.tensor(masked_node_indices)
        batched_data.y = batched_data.x[masked_node_indices][:, 0]

        # mask nodes' features
        for node_idx in masked_node_indices:
            batched_data.x[node_idx] = torch.tensor([self.mask_atom_id - 1] + [0] * (batched_data.x.shape[1] - 1))
    
        # generate predictions
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        batched_data.x = h_node
        prediction_class = self.predictor(batched_data)[masked_node_indices]
        print('prediction_class', prediction_class.max(), prediction_class.min())
        print('batched_data.y', batched_data.y.max(), batched_data.y.min())

        
        # target_class = batched_data.y.to(torch.float32)
        loss_class = class_criterion(prediction_class.to(torch.float32), batched_data.y.long())
        
        fingerprint_loss = get_fingerprint_loss(batched_data.smiles, h_rep)
                
        loss = self.lw_rec * loss_class + (1 - self.lw_rec) * fingerprint_loss

        return loss

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        return {"graph": h_rep, "node": h_node}
    
class GNN_Decoder(torch.nn.Module):
    def __init__(self, hidden_size, out_dim, JK = "last", gnn_name = "gin"):
        super().__init__()
        if gnn_name == 'gin':
            self.conv = GINConv(hidden_size)
        elif gnn_name == 'gcn':
            self.conv = GCNConv(hidden_size)
        else:
            raise ValueError('Undefined GNN type called {}'.format(gnn_name))
        self.dec_token = torch.nn.Parameter(torch.zeros([1, hidden_size]))
        self.enc_to_dec = torch.nn.Linear(hidden_size, hidden_size, bias=False)    
        self.activation = torch.nn.PReLU()
        self.out = torch.nn.Linear(hidden_size, out_dim) 

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        masked_node_indices = batched_data.masked_node_indices

        x = self.activation(x)
        x = self.enc_to_dec(x)
        x[masked_node_indices] = self.dec_token.detach().expand(len(masked_node_indices), -1)

        x = self.conv(x, edge_index, edge_attr)
        out = self.out(x)
        return out