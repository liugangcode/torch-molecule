import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data

from ...nn import GNN_node, GNN_node_Virtualnode
from ...utils import init_weights

from .utils import ExtractSubstructureContextPair

class_criterion = torch.nn.CrossEntropyLoss()

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class GNN(nn.Module):
    def __init__(
        self,
        num_layer,
        hidden_size,
        drop_ratio=0.5,
        norm_layer="batch_norm",
        encoder_type="gin-virtual",
        readout="max",
        mode="cbow",
        context_size=3,
        neg_samples=1
    ):
        super(GNN, self).__init__()
        gnn_name = encoder_type.split("-")[0]
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.mode = mode
        self.neg_samples = neg_samples
        self.context_size = context_size

        encoder_params_substruct = {
            "num_layer": num_layer,
            "hidden_size": hidden_size,
            "JK": "last", 
            "drop_ratio": drop_ratio,
            "residual": True,
            "gnn_name": gnn_name,
            "norm_layer": norm_layer
        }
        
        encoder_params_context = {
            "num_layer": context_size,
            "hidden_size": hidden_size,
            "JK": "last", 
            "drop_ratio": drop_ratio,
            "residual": True,
            "gnn_name": gnn_name,
            "norm_layer": norm_layer
        }
        
        # Choose encoder type based on encoder_model
        encoder_class = GNN_node_Virtualnode if "virtual" in encoder_type else GNN_node
        self.graph_encoder_substuct = encoder_class(**encoder_params_substruct)
        pooling_funcs = {
            "sum": global_add_pool,
            "mean": global_mean_pool,
            "max": global_max_pool
        }
        self.pool = pooling_funcs.get(readout)
        if self.pool is None:
            raise ValueError(f"Invalid graph pooling type {readout}.")

        self.graph_encoder_context = encoder_class(**encoder_params_context)
    
    def initialize_parameters(self, seed=None):
        """
        Randomly initialize all model parameters using the init_weights function.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize the main components
        init_weights(self.graph_encoder_substuct)
        init_weights(self.graph_encoder_context)
        
        # Reset all parameters using PyTorch Geometric's reset function
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                init_weights(module)
        
        self.apply(reset_parameters)

    def compute_loss(self, batched_data):
        device = batched_data.x.device
        extract_context = ExtractSubstructureContextPair(self.num_layer, self.num_layer - 1, self.num_layer + self.context_size - 1)
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]
        for key in keys:
            batched_data[key] = []

        #used for pooling the substructure
        batched_data.batch_substruct = [] 
        #used for pooling the context
        batched_data.batch_context = [] 
        batched_data.batch_overlapped_context = []
        batched_data.overlapped_context_size = []
        
        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        
        for j in range(len(batched_data)):
            data = batched_data[j]
            data = extract_context(data)

            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                batched_data.batch_substruct.append(torch.full((len(data.substruct_node_idxes), ), i, dtype=torch.long).to(device))
                batched_data.batch_context.append(torch.full((len(data.context_node_idxes), ), i, dtype=torch.long).to(device))
                batched_data.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long).to(device))
                batched_data.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"] else item
                    batched_data[key].append(item.to(device))
                

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"] else item
                    batched_data[key].append(item.to(device))

                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct   
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            if key in ["edge_index", "edge_index_substruct", "edge_index_context"]:
                dim = -1
            else:
                dim = 0
            batched_data[key] = torch.cat(batched_data[key], dim=dim)

        batched_data.batch_substruct = torch.cat(batched_data.batch_substruct, dim=-1).to(device)
        batched_data.batch_context = torch.cat(batched_data.batch_context, dim=-1).to(device)
        batched_data.batch_overlapped_context = torch.cat(batched_data.batch_overlapped_context, dim=-1).to(device)
        batched_data.overlapped_context_size = torch.LongTensor(batched_data.overlapped_context_size).to(device)

        # generate predictions
        substruct_data = Data(x=batched_data.x_substruct, edge_index=batched_data.edge_index_substruct, edge_attr=batched_data.edge_attr_substruct, batch=batched_data.batch_substruct)
        context_data = Data(x=batched_data.x_context, edge_index=batched_data.edge_index_context, edge_attr=batched_data.edge_attr_context, batch=batched_data.batch_context)
        
        substruct_node_rep, _ = self.graph_encoder_substuct(substruct_data)
        #substruct_h_rep = self.pool(substruct_node_rep, batched_data.batch_substruct)
        
        context_node_rep, _ = self.graph_encoder_context(context_data)
        context_h_rep = self.pool(context_node_rep, batched_data.batch_context)
        
        # contexts are represented by 
        if self.mode == "cbow":
            # positive context representation
            context_rep = self.pool(context_node_rep[batched_data.overlap_context_substruct_idx], batched_data.batch_overlapped_context)
            # negative contexts are obtained by shifting the indicies of context embeddings
            neg_context_rep = torch.cat([context_rep[cycle_index(len(context_rep), i+1)] for i in range(self.neg_samples)], dim = 0)
            
            pred_pos = torch.sum(substruct_node_rep[batched_data.center_substruct_idx] * context_rep, dim = 1)
            pred_neg = torch.sum(substruct_node_rep[batched_data.center_substruct_idx].repeat((self.neg_samples, 1))*neg_context_rep, dim = 1)

        elif self.mode == "skipgram":

            expanded_substruct_rep = torch.cat([substruct_node_rep[batched_data.center_substruct_idx][i].repeat((batched_data.overlapped_context_size[i],1)) for i in range(len(substruct_node_rep[batched_data.center_substruct_idx]))], dim = 0)
            pred_pos = torch.sum(expanded_substruct_rep * context_h_rep, dim = 1)

            #shift indices of substructures to create negative examples
            shifted_expanded_substruct_rep = []
            for i in range(self.neg_samples):
                shifted_substruct_rep = substruct_node_rep[batched_data.center_substruct_idx][cycle_index(len(substruct_node_rep[batched_data.center_substruct_idx]), i+1)]
                shifted_expanded_substruct_rep.append(torch.cat([shifted_substruct_rep[i].repeat((batched_data.overlapped_context_size[i],1)) for i in range(len(shifted_substruct_rep))], dim = 0))

            shifted_expanded_substruct_rep = torch.cat(shifted_expanded_substruct_rep, dim = 0)
            pred_neg = torch.sum(shifted_expanded_substruct_rep * context_h_rep.repeat((self.neg_samples, 1)), dim = 1)

        else:
            raise ValueError("Invalid mode!")
        
        loss_pos = class_criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = class_criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())
        
        loss_class = loss_pos + self.neg_samples*loss_neg

        return loss_class

    def forward(self, batched_data):
        h_node, _ = self.graph_encoder_substuct(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        return {"graph": h_rep, "node": h_node}