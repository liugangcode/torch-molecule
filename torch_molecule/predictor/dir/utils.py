import torch
from torch_geometric.utils import degree
from torch_geometric.nn.conv import MessagePassing

def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

def relabel(x, edge_index, batch, pos=None, keep_all_nodes=False):
    num_nodes = x.size(0)
    
    if keep_all_nodes or edge_index.size(1) == 0:
        # Keep all nodes to maintain consistent batch indexing
        sub_nodes = torch.arange(num_nodes, device=x.device)
        # If no edges, return original data with empty edge_index
        if edge_index.size(1) == 0:
            empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            return x, empty_edge_index, batch, pos
    else:
        sub_nodes = torch.unique(edge_index)
    
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    
    if edge_index.size(1) > 0:
        row, col = edge_index
        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def set_masks(mask: torch.Tensor, model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.explain = True
            module._edge_mask = mask
        
def clear_masks(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.explain = None
            module._edge_mask = None