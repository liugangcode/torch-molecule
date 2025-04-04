import os
import torch
import torch.nn as nn
# from torch_scatter import scatter_add
from .utils import scatter_add

from ...nn import GNN_node, GNN_node_Virtualnode, MLP
from ...utils import init_weights

class GREA(nn.Module):
    def __init__(
        self,
        num_task,
        num_layer,
        gamma=0.4,
        hidden_size=300,
        gnn_type="gin-virtual",
        drop_ratio=0.5,
        norm_layer="batch_norm",
        augmented_feature=['maccs', 'morgan']
    ):
        super(GREA, self).__init__()
        gnn_name = gnn_type.split("-")[0]
        self.num_task = num_task
        self.gamma = gamma
        self.hidden_size = hidden_size

        if "virtual" in gnn_type:
            rationale_encoder = GNN_node_Virtualnode(
                2, hidden_size, JK="last", drop_ratio=drop_ratio, residual=True, gnn_name=gnn_name
            )
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
            rationale_encoder = GNN_node(
                2, hidden_size, JK="last", drop_ratio=drop_ratio, residual=True, gnn_name=gnn_name
            )
            self.graph_encoder = GNN_node(
                num_layer,
                hidden_size,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )

        self.separator = Separator(
            rationale_encoder=rationale_encoder,
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 2 * hidden_size),
                torch.nn.BatchNorm1d(2 * hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(2 * hidden_size, 1),
            ),
        )
        
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
        init_weights(self.separator)
        init_weights(self.predictor)
        
        # Reset all parameters using PyTorch Geometric's reset function
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                init_weights(module)
        
        self.apply(reset_parameters)

    def _augment_graph_features(self, batched_data, h_r, h_rep):
        if self.augmented_feature:
            if 'morgan' in self.augmented_feature:
                morgan = batched_data.morgan.type_as(h_r)
                h_r = torch.cat((h_r, morgan), dim=1)
                morgans = morgan.repeat_interleave(batched_data.batch[-1]+1, dim=0)
                h_rep = torch.cat((h_rep, morgans), dim=1)
            if 'maccs' in self.augmented_feature:
                maccs = batched_data.maccs.type_as(h_r)
                h_r = torch.cat((h_r, maccs), dim=1)
                maccses = maccs.repeat_interleave(batched_data.batch[-1]+1, dim=0)
                h_rep = torch.cat((h_rep, maccses), dim=1)
        return h_r, h_rep
    
    def compute_loss(self, batched_data, criterion):
        h_node, _ = self.graph_encoder(batched_data)
        h_r, h_env, rationale_size, envir_size, _ = self.separator(batched_data, h_node)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.hidden_size)
        h_r, h_rep = self._augment_graph_features(batched_data, h_r, h_rep)
        pred_rem = self.predictor(h_r)
        pred_rep = self.predictor(h_rep)
        loss = torch.abs(
            rationale_size / (rationale_size + envir_size)
            - self.gamma * torch.ones_like(rationale_size)
        ).mean()
        target = batched_data.y.to(torch.float32)
        is_labeled = batched_data.y == batched_data.y
        loss += criterion(pred_rem.to(torch.float32)[is_labeled], target[is_labeled]).mean()
        target_rep = batched_data.y.to(torch.float32).repeat_interleave(
            batched_data.batch[-1] + 1, dim=0
        )
        is_labeled_rep = target_rep == target_rep
        loss += criterion(pred_rep.to(torch.float32)[is_labeled_rep], target_rep[is_labeled_rep]).mean()
        return loss

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_r, h_env, _, _, node_score = self.separator(batched_data, h_node)
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.hidden_size)
        h_r, h_rep = self._augment_graph_features(batched_data, h_r, h_rep)
        prediction = self.predictor(h_r)
        pred_rep = self.predictor(h_rep).view(h_r.size(0), -1)
        if pred_rep.size(1) > 1:
            variance = pred_rep.var(dim=-1, keepdim=True)
        else:
            variance = torch.zeros_like(pred_rep)
        num_graphs = batched_data.batch.max().item() + 1
        score_by_graph = [node_score[batched_data.batch == i].view(-1).tolist() for i in range(num_graphs)]
        return {"prediction": prediction, "variance": variance, "score": score_by_graph, "representation": h_r}

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
