import os
import torch
import torch.nn as nn
from torch_scatter import scatter_add

from ..components.gnn_components import GNN_node, GNN_node_Virtualnode
from ...utils import init_weights

class GREA(nn.Module):
    def __init__(
        self,
        num_tasks,
        num_layer,
        gamma=0.4,
        emb_dim=300,
        gnn_type="gin-virtual",
        drop_ratio=0.5,
        norm_layer="batch_norm",
    ):
        super(GREA, self).__init__()
        gnn_name = gnn_type.split("-")[0]
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.emb_dim = emb_dim

        if "virtual" in gnn_type:
            rationale_encoder = GNN_node_Virtualnode(
                2, emb_dim, JK="last", drop_ratio=drop_ratio, residual=True, gnn_name=gnn_name
            )
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
            rationale_encoder = GNN_node(
                2, emb_dim, JK="last", drop_ratio=drop_ratio, residual=True, gnn_name=gnn_name
            )
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )

        self.separator = Separator(
            rationale_encoder=rationale_encoder,
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(2 * emb_dim, 1),
            ),
        )
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
        init_weights(self.separator)
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
        h_r, h_env, rationale_size, envir_size, _ = self.separator(batched_data, h_node)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.emb_dim)
        pred_rem = self.predictor(h_r)
        pred_rep = self.predictor(h_rep)
        loss = torch.abs(
            rationale_size / (rationale_size + envir_size)
            - self.gamma * torch.ones_like(rationale_size)
        ).mean()

        target = batched_data.y.to(torch.float32)
        is_labeled = batched_data.y == batched_data.y
        loss += criterion(pred_rem.to(torch.float32)[is_labeled], target[is_labeled])
        target_rep = batched_data.y.to(torch.float32).repeat_interleave(
            batched_data.batch[-1] + 1, dim=0
        )
        is_labeled_rep = target_rep == target_rep
        loss += criterion(pred_rep.to(torch.float32)[is_labeled_rep], target_rep[is_labeled_rep])

        return loss

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_r, h_env, _, _, node_score = self.separator(batched_data, h_node)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.emb_dim)
        prediction = self.predictor(h_r)
        variance = self.predictor(h_rep).view(h_r.size(0), -1).var(dim=-1, keepdim=True)
        num_graphs = batched_data.batch.max().item() + 1
        score_by_graph = [node_score[batched_data.batch == i].view(-1).tolist() for i in range(num_graphs)]
        return {"prediction": prediction, "variance": variance, "score": score_by_graph}

class Separator(torch.nn.Module):
    def __init__(self, rationale_encoder, gate_nn, nn=None):
        super(Separator, self).__init__()
        self.rationale_encoder = rationale_encoder
        self.gate_nn = gate_nn
        self.nn = nn

    def forward(self, batched_data, h_node, size=None):
        x, _ = self.rationale_encoder(batched_data)
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)

        h_out = scatter_add(gate * h_node, batch, dim=0, dim_size=size)
        c_out = scatter_add((1 - gate) * h_node, batch, dim=0, dim_size=size)

        rationale_size = scatter_add(gate, batch, dim=0, dim_size=size)
        envir_size = scatter_add((1 - gate), batch, dim=0, dim_size=size)

        return h_out, c_out, rationale_size + 1e-8, envir_size + 1e-8, gate
