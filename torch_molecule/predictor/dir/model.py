import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from .utils import split_batch, relabel, set_masks, clear_masks
from ...nn import GNN_node, GNN_node_Virtualnode
from ...utils import init_weights

class DIR(nn.Module):
    def __init__(
        self,
        num_task,
        num_layer,
        causal_ratio=0.8,
        hidden_size=300,
        gnn_type="gin-virtual",
        drop_ratio=0.5,
        norm_layer="batch_norm",
        graph_pooling="sum",
        augmented_feature=['maccs', 'morgan']
    ):
        super(DIR, self).__init__()
        gnn_name = gnn_type.split("-")[0]
        self.num_task = num_task
        self.hidden_size = hidden_size
        self.causal_ratio = causal_ratio
        if "virtual" in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                hidden_size,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
                encode_atom=False
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
                encode_atom=False
            )

        self.causal_att_net = CausalAttNet(causal_ratio=causal_ratio, hidden_size=hidden_size)
        
        graph_dim = hidden_size
        self.augmented_feature = augmented_feature
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Invalid graph pooling type {graph_pooling}.")
        
        if augmented_feature:
            if "morgan" in augmented_feature:
                graph_dim += 1024
            if "maccs" in augmented_feature:
                graph_dim += 167
        
        self.causal_lin = torch.nn.Linear(graph_dim, num_task)
        self.conf_lin = torch.nn.Linear(hidden_size, num_task)
    
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        h_node, _ = self.graph_encoder(x, edge_index, edge_attr, batch)
        h_graph = self.pool(h_node, batch)
        return h_graph
    
    def get_causal_pred(self, batched_data, h_graph):
        h_graph = self._augment_graph_features(batched_data, h_graph)
        return self.causal_lin(h_graph)
    
    def get_conf_pred(self, conf_graph_x):
        return self.conf_lin(conf_graph_x)
    
    def get_comb_pred(self, batched_data, causal_graph_x, conf_graph_x):
        causal_graph_x = self._augment_graph_features(batched_data, causal_graph_x)
        causal_pred = self.causal_lin(causal_graph_x)
        conf_pred = self.conf_lin(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred

    def initialize_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize the main components
        init_weights(self.graph_encoder)
        init_weights(self.causal_att_net)
        init_weights(self.causal_lin)
        init_weights(self.conf_lin)
        
        # Reset all parameters using PyTorch Geometric's reset function
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                init_weights(module)
        
        self.apply(reset_parameters)

    def _augment_graph_features(self, batched_data, h):
        if self.augmented_feature:
            if 'morgan' in self.augmented_feature:
                morgan = batched_data.morgan.type_as(h)
                h = torch.cat((h, morgan), dim=1)
            if 'maccs' in self.augmented_feature:
                maccs = batched_data.maccs.type_as(h)
                h = torch.cat((h, maccs), dim=1)
        return h
    
    def compute_loss(self, batched_data, criterion, alpha_prime):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
        (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), pred_edge_weight = self.causal_att_net(batched_data)
            
        set_masks(causal_edge_weight, self.graph_encoder)
        set_masks(causal_edge_weight, self.causal_lin)
        set_masks(causal_edge_weight, self.conf_lin)
        causal_rep = self.get_graph_rep(x=causal_x, edge_index=causal_edge_index, edge_attr=causal_edge_attr, batch=causal_batch)
        causal_out = self.get_causal_pred(batched_data, causal_rep)
        clear_masks(self.graph_encoder)
        clear_masks(self.causal_lin)
        clear_masks(self.conf_lin)

        set_masks(conf_edge_weight, self.graph_encoder)
        set_masks(conf_edge_weight, self.causal_lin)
        set_masks(conf_edge_weight, self.conf_lin)
        conf_rep = self.get_graph_rep(x=conf_x, edge_index=conf_edge_index, edge_attr=conf_edge_attr, batch=conf_batch).detach()
        conf_out = self.get_conf_pred(conf_rep)
        clear_masks(self.graph_encoder)
        clear_masks(self.causal_lin)
        clear_masks(self.conf_lin)

        batched_data.y = batched_data.y[0:causal_out.size(0),:]
        is_labeled = batched_data.y == batched_data.y
        
        causal_loss = criterion(
            causal_out.to(torch.float32)[is_labeled], 
            batched_data.y.to(torch.float32)[is_labeled]
            ).mean()
        conf_loss = criterion(
            conf_out.to(torch.float32)[is_labeled], 
            batched_data.y.to(torch.float32)[is_labeled]
            ).mean()

        env_loss = torch.tensor([]).to(batched_data.y.device)
        for conf in conf_rep:
            rep_out = self.get_comb_pred(batched_data, causal_rep, conf)
            tmp = criterion(rep_out.to(torch.float32)[is_labeled], batched_data.y.to(torch.float32)[is_labeled]).mean()
            env_loss = torch.cat([env_loss, tmp.unsqueeze(0)])
        causal_loss += alpha_prime * env_loss.mean()
        env_loss = alpha_prime * torch.var(env_loss * conf_rep.size(0))
        
        return causal_loss, conf_loss, env_loss

    def forward(self, batched_data):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), _, _ = self.causal_att_net(batched_data)
        set_masks(causal_edge_weight, self.graph_encoder)
        set_masks(causal_edge_weight, self.causal_lin)
        set_masks(causal_edge_weight, self.conf_lin)

        causal_rep = self.get_graph_rep(x=causal_x, edge_index=causal_edge_index, edge_attr=causal_edge_attr, batch=causal_batch)
        causal_out = self.get_causal_pred(batched_data, causal_rep)
        clear_masks(self.graph_encoder)
        clear_masks(self.causal_lin)
        clear_masks(self.conf_lin)

        return {"prediction": causal_out, "representation": causal_rep}

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)
    
    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)

class CausalAttNet(nn.Module):
    def __init__(self, causal_ratio, hidden_size):
        super(CausalAttNet, self).__init__()
        self.gnn_node = GNN_node_Virtualnode(2, hidden_size, JK="last", drop_ratio=0., residual=True, gnn_name="gin", norm_layer="batch_norm")
        self.linear = nn.Linear(hidden_size*2, 1)
        self.ratio = causal_ratio
        
    def forward(self, data):
        x, _ = self.gnn_node(data.x, data.edge_index, data.edge_attr, data.batch)

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        pred_edge_weight = self.linear(edge_rep).view(-1)
        causal_edge_index = torch.LongTensor([[],[]]).to(x.device)
        causal_edge_weight = torch.tensor([]).to(x.device)
        causal_edge_attr = torch.LongTensor([]).to(x.device)
        conf_edge_index = torch.LongTensor([[],[]]).to(x.device)
        conf_edge_weight = torch.tensor([]).to(x.device)
        conf_edge_attr = torch.LongTensor([]).to(x.device)

        edge_indices, _, _, num_edges, cum_edges = split_batch(data)
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
            n_reserve =  int(self.ratio * N)
            edge_attr = data.edge_attr[C:C+N]
            single_mask = pred_edge_weight[C:C+N]
            single_mask_detach = pred_edge_weight[C:C+N].detach().cpu().numpy()
            rank = np.argpartition(-single_mask_detach, n_reserve)
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)
            causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
            conf_edge_weight = torch.cat([conf_edge_weight, -1 * single_mask[idx_drop]])
            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch, keep_all_nodes=True)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch, keep_all_nodes=True)
        
        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),\
                pred_edge_weight