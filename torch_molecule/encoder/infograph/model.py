import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ...nn import GNN_node, GNN_node_Virtualnode

from .module import PriorDiscriminator, FF
from .loss_function import local_global_loss_

class GNN(nn.Module):
    def __init__(
        self,
        num_layer,
        embedding_dim,
        drop_ratio=0.5,
        norm_layer="batch_norm",
        encoder_type="gin-virtual",
        readout="add",
        use_prior=False,
    ):
        super(GNN, self).__init__()
        gnn_name = encoder_type.split("-")[0]

        self.hidden_size = embedding_dim // num_layer
        self.embedding_dim = embedding_dim

        encoder_params = {
            "num_layer": num_layer,
            "hidden_size": self.hidden_size,
            "JK": "last", 
            "drop_ratio": drop_ratio,
            "residual": False,
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

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        self.use_prior = use_prior
        if self.use_prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def compute_loss(self, batched_data, lw_prior=0.1):
        _, h_node_list = self.graph_encoder(batched_data)
        h_node_list = h_node_list[1:]
        hpool = [self.pool(h_node, batched_data.batch) for h_node in h_node_list]
        h_g = torch.cat(hpool, dim=1)
        h_l = torch.cat(h_node_list, dim=1)

        g_enc = self.global_d(h_g)
        l_enc = self.local_d(h_l)

        local_global_loss = local_global_loss_(l_enc, g_enc, batched_data.batch, "JSD")

        if self.use_prior:
            prior = torch.rand_like(h_g)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(h_g)).mean()
            PRIOR = - (term_a + term_b) * lw_prior
        else:
            PRIOR = torch.tensor(0.0)
        
        return local_global_loss, PRIOR

    def forward(self, batched_data):
        _, h_node_list = self.graph_encoder(batched_data)
        h_node_list = h_node_list[1:]
        hpool = [self.pool(h_node, batched_data.batch) for h_node in h_node_list]
        h_g = torch.cat(hpool, dim=1)

        return {"graph": h_g}
