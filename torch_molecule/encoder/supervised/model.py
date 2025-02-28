import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ...nn import GNN_node, GNN_node_Virtualnode, MLP
from ...utils import init_weights

reg_criterion = torch.nn.L1Loss()
class_criterion = torch.nn.BCEWithLogitsLoss()

class GNN(nn.Module):
    def __init__(
        self,
        num_layer,
        hidden_size,
        num_task,
        drop_ratio=0.5,
        norm_layer="batch_norm",
        encoder_type="gin-virtual",
        readout="max",
    ):
        super(GNN, self).__init__()
        gnn_name = encoder_type.split("-")[0]
        self.num_task = num_task
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
        
        # Choose encoder type based on encoder
        encoder_class = GNN_node_Virtualnode if "virtual" in encoder_type else GNN_node
        self.graph_encoder = encoder_class(**encoder_params)
        pooling_funcs = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool
        }
        self.readout = pooling_funcs.get(readout)
        if self.readout is None:
            raise ValueError(f"Invalid graph pooling type {readout}.")

        self.predictor = MLP(hidden_size, hidden_features=2 * hidden_size, out_features=num_task)
    
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

    def compute_loss(self, batched_data, is_class):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.readout(h_node, batched_data.batch)
        prediction = self.predictor(h_rep)

        target = batched_data.y.to(torch.float32)
        is_labeled = ~torch.isnan(target)
        is_labeled_class = is_labeled[:, is_class]
        is_labeled_reg = is_labeled[:, ~is_class]

        target_class = target[:, is_class]
        target_reg = target[:, ~is_class]
        prediction_class = prediction[:, is_class]
        prediction_reg = prediction[:, ~is_class]
        
        loss_class = class_criterion(prediction_class.to(torch.float32)[is_labeled_class], target_class[is_labeled_class])
        loss_reg = reg_criterion(prediction_reg.to(torch.float32)[is_labeled_reg], target_reg[is_labeled_reg])
        loss = loss_class + loss_reg
        return loss

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.readout(h_node, batched_data.batch)
        return {"graph": h_rep, "node": h_node}