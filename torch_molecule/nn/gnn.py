import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn.norm import GraphNorm, PairNorm, DiffGroupNorm, InstanceNorm, LayerNorm, GraphSizeNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
try:
    from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_add
    _has_torch_scatter = True
except ImportError:
    scatter_min = scatter_max = scatter_mean = scatter_add = None
    _has_torch_scatter = False

from ..utils import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(torch.nn.Module):
    """Encodes atom features into a fixed-size vector representation.
    
    This module converts categorical atom features into embeddings and combines them
    to create a unified atom representation.
    
    Parameters
    ----------
    hidden_size : int
        Dimensionality of the output atom embedding vectors.
        
    Notes
    -----
    Each atom feature is embedded separately using an Embedding layer, then
    these embeddings are summed to produce the final representation.
    The embedding weights are initialized using Xavier uniform initialization
    with max_norm=1 constraint.
    """
    def __init__(self, hidden_size):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, hidden_size, max_norm=1)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        """Transform atom features into embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape [num_atoms, num_features] containing categorical
            atom features.
            
        Returns
        -------
        torch.Tensor
            Atom embeddings of shape [num_atoms, hidden_size].
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

class BondEncoder(torch.nn.Module):
    """Encodes bond features into a fixed-size vector representation.
    
    This module converts categorical bond features into embeddings and combines them
    to create a unified bond representation.
    
    Parameters
    ----------
    hidden_size : int
        Dimensionality of the output bond embedding vectors.
        
    Notes
    -----
    Each bond feature is embedded separately using an Embedding layer, then
    these embeddings are summed to produce the final representation.
    The embedding weights are initialized using Xavier uniform initialization
    with max_norm=1 constraint.
    """
    def __init__(self, hidden_size):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, hidden_size, max_norm=1)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        """Transform bond features into embeddings.
        
        Parameters
        ----------
        edge_attr : torch.Tensor
            Tensor of shape [num_bonds, num_features] containing categorical
            bond features.
            
        Returns
        -------
        torch.Tensor
            Bond embeddings of shape [num_bonds, hidden_size].
        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding

class GINConv(MessagePassing):
    def __init__(self, hidden_size, output_size=None):
        '''
            hidden_size (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "add")
        if output_size is None:
            output_size = hidden_size

        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden_size, 2*hidden_size), torch.nn.BatchNorm1d(2*hidden_size), torch.nn.ReLU(), torch.nn.Linear(2*hidden_size, output_size))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(hidden_size = hidden_size)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
    
class GCNConv(MessagePassing):
    def __init__(self, hidden_size, output_size=None):
        super(GCNConv, self).__init__(aggr='add')
        if output_size is None:
            output_size = hidden_size

        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.root_emb = torch.nn.Embedding(1, output_size)
        self.bond_encoder = BondEncoder(hidden_size = output_size)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index
        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GINConv_BF(MessagePassing):
    def __init__(self, hidden_size, output_size=None):
        super().__init__(aggr=None) 
        if output_size is None:
            output_size = hidden_size
        self.f_agg = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_size, 2*hidden_size),
            torch.nn.BatchNorm1d(2*hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_size, output_size),
        )
        self.f_up = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_size, 2*hidden_size),
            torch.nn.BatchNorm1d(2*hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_size, output_size),
        )
        self.bond_encoder = BondEncoder(hidden_size=output_size)

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.bond_encoder(edge_attr)
        agg = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        out = self.f_up(torch.cat([agg, x], dim=1))
        return out

    def message(self, x_j, edge_attr):
        return self.f_agg(torch.cat([x_j, edge_attr], dim=1))

    def aggregate(self, inputs, index, dim_size=None):
        if not _has_torch_scatter or scatter_min is None:
            raise ImportError("BFGNN requires `torch_scatter` package. Please install it via `pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`.")

        out, _ = scatter_min(inputs, index, dim=0, dim_size=dim_size)
        out[out == float('inf')] = 0.0
        return out

class GCNConv_BF(MessagePassing):
    def __init__(self, hidden_size, output_size=None):
        super().__init__(aggr=None)  
        if output_size is None:
            output_size = hidden_size
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.f_agg = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_size, 2*hidden_size),
            torch.nn.BatchNorm1d(2*hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_size, output_size),
        )
        self.f_up = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_size, 2*hidden_size),
            torch.nn.BatchNorm1d(2*hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_size, output_size),
        )
        self.bond_encoder = BondEncoder(hidden_size=output_size)

    def forward(self, x, edge_index, edge_attr):
        x_lin = self.linear(x)

        row, col = edge_index
        deg = degree(row, x_lin.size(0), dtype=x_lin.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_emb = self.bond_encoder(edge_attr)
        agg = self.propagate(
            edge_index,
            x=x_lin,
            edge_attr=edge_emb,
            norm=norm,       
            size=None
        )
        return self.f_up(torch.cat([agg, x_lin], dim=1))

    def message(self, x_j, edge_attr, norm):
        m = self.f_agg(torch.cat([x_j, edge_attr], dim=1))
        return norm.view(-1, 1) * F.relu(m)

    def aggregate(self, inputs, index, dim_size=None):
        if not _has_torch_scatter or scatter_min is None:
            raise ImportError("BFGNN requires `torch_scatter` package. Please install it via `pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`.")

        out, _ = scatter_min(inputs, index, dim=0, dim_size=dim_size)
        out[out == float('inf')] = 0.0
        return out

class GINConv_GRIN(MessagePassing):
    def __init__(self, hidden_size, output_size=None):
        super().__init__(aggr=None) 
        if output_size is None:
            output_size = hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_size, 2*hidden_size),
            torch.nn.BatchNorm1d(2*hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_size, output_size),
        )
        self.eps = torch.nn.Parameter(torch.zeros(1))
        self.bond_encoder = BondEncoder(hidden_size=output_size)
        self.root_emb = torch.nn.Embedding(1, output_size)

    def forward(self, x, edge_index, edge_attr, visited):
        edge_emb = self.bond_encoder(edge_attr)
        row, _ = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1

        agg = self.propagate(edge_index, x=x, edge_attr=edge_emb)

        agg[visited] = 0.0

        concat = torch.cat([x, agg], dim=1)
        out = self.mlp(concat)
        out = out + F.relu((1 + self.eps) * x + self.root_emb.weight) * (1.0 / deg.view(-1,1))

        _, selected = agg.max(dim=0)
        new_visited = visited.clone()
        new_visited[selected] = True
        mask = torch.zeros_like(visited)
        mask[selected] = True
        out = torch.where(mask.unsqueeze(1), out, x)

        return out, new_visited

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def aggregate(self, inputs, index, dim_size=None):
        if not _has_torch_scatter or scatter_max is None:
            raise ImportError("GRIN requires `torch_scatter` package. Please install it via `pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`.")

        out, _ = scatter_max(inputs, index, dim=0, dim_size=dim_size)
        out[out == float('inf')] = 0.0
        return out

class GCNConv_GRIN(MessagePassing):
    def __init__(self, hidden_size, output_size=None):
        super().__init__(aggr=None)  
        if output_size is None:
            output_size = hidden_size
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.root_emb = torch.nn.Embedding(1, output_size)
        self.bond_encoder = BondEncoder(hidden_size=output_size)
        self.f_up = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_size, 2*hidden_size),
            torch.nn.BatchNorm1d(2*hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(2*hidden_size, output_size),
        )

    def forward(self, x, edge_index, edge_attr, visited):
        x_lin = self.linear(x)
        row, col = edge_index
        deg = degree(row, x_lin.size(0), dtype=x_lin.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_emb = self.bond_encoder(edge_attr)
        agg = self.propagate(
            edge_index,
            x=x_lin,
            edge_attr=edge_emb,
            norm=norm,
        )

        agg[visited] = 0.0

        concat = torch.cat([x_lin, agg], dim=1)
        out = self.f_up(concat)
        out = out + F.relu(x_lin + self.root_emb.weight) * (1.0 / deg.view(-1,1))

        _, selected = agg.max(dim=0)
        new_visited = visited.clone()
        new_visited[selected] = True
        mask = torch.zeros_like(visited)
        mask[selected] = True
        out = torch.where(mask.unsqueeze(1), out, x)

        return out, new_visited

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1,1) * F.relu(x_j + edge_attr)

    def aggregate(self, inputs, index, dim_size=None):
        if not _has_torch_scatter or scatter_max is None:
            raise ImportError("GRIN requires `torch_scatter` package. Please install it via `pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`.")

        out, _ = scatter_max(inputs, index, dim=0, dim_size=dim_size)
        out[out == float('inf')] = 0.0
        return out

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, hidden_size, drop_ratio = 0.5, JK = "last", residual = False, gnn_name = 'gin', norm_layer = 'batch_norm', encode_atom = True, algorithm_aligned = None):
        '''
            hidden_size (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.norm_layer = norm_layer
        self.encode_atom = encode_atom
        self.algorithm_aligned = algorithm_aligned
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(hidden_size)
        self.bond_encoder = BondEncoder(hidden_size)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == 'gin':
                if algorithm_aligned == 'bf':
                    self.convs.append(GINConv_BF(hidden_size))
                elif algorithm_aligned == 'mst':
                    self.convs.append(GINConv_GRIN(hidden_size))
                else:
                    self.convs.append(GINConv(hidden_size))
            elif gnn_name == 'gcn':
                if algorithm_aligned == 'bf':
                    self.convs.append(GCNConv_BF(hidden_size))
                elif algorithm_aligned == 'mst':
                    self.convs.append(GCNConv_GRIN(hidden_size))
                else:
                    self.convs.append(GCNConv(hidden_size))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_name))

            if norm_layer.split('_')[0] == 'batch':
                if norm_layer.split('_')[-1] == 'notrack':
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_size, track_running_stats=False, affine=False))
                else:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_size))
            elif norm_layer.split('_')[0] == 'instance':
                self.batch_norms.append(InstanceNorm(hidden_size))
            elif norm_layer.split('_')[0] == 'layer':
                self.batch_norms.append(LayerNorm(hidden_size))
            elif norm_layer.split('_')[0] == 'graph':
                self.batch_norms.append(GraphNorm(hidden_size))
            elif norm_layer.split('_')[0] == 'size':
                self.batch_norms.append(GraphSizeNorm())
            elif norm_layer.split('_')[0] == 'pair':
                self.batch_norms.append(PairNorm(hidden_size))
            elif norm_layer.split('_')[0] == 'group':
                self.batch_norms.append(DiffGroupNorm(hidden_size, groups=4))
            else:
                raise ValueError('Undefined normalization layer called {}'.format(norm_layer))
        if norm_layer.split('_')[1] == 'size':
            self.graph_size_norm = GraphSizeNorm()

    # def forward(self, batched_data):
    #     x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
    def forward(self, *args):
        if len(args) == 1:
            # Case 1: batched_data input
            batched_data = args[0]
            x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        elif len(args) == 4:
            # Case 2: separate inputs
            x, edge_index, edge_attr, batch = args
        else:
            raise ValueError("forward expects either 1 batched_data argument or 4 separate arguments (x, edge_index, edge_attr, batch)")
        
        if self.encode_atom:
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]
        if self.algorithm_aligned == 'mst':
            visited = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for layer in range(self.num_layer):
            if self.algorithm_aligned == 'mst':
                h, visited = self.convs[layer](h_list[layer], edge_index, edge_attr, visited)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            if self.norm_layer.split('_')[0] == 'batch':
                h = self.batch_norms[layer](h)
            else:
                h = self.batch_norms[layer](h, batch)
            if self.norm_layer.split('_')[1] == 'size':
                h = self.graph_size_norm(h, batch)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.relu(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)
            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation, h_list

### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, hidden_size, drop_ratio = 0.5, JK = "last", residual = False, gnn_name = 'gin', norm_layer = 'batch_norm', encode_atom = True, algorithm_aligned = None):
        '''
            hidden_size (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.norm_layer = norm_layer
        self.encode_atom = encode_atom
        self.algorithm_aligned = algorithm_aligned
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(hidden_size)
        self.bond_encoder = BondEncoder(hidden_size)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, hidden_size)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_name == 'gin':
                if algorithm_aligned == 'bf':
                    self.convs.append(GINConv_BF(hidden_size))
                elif algorithm_aligned == 'mst':
                    self.convs.append(GINConv_GRIN(hidden_size))
                else:
                    self.convs.append(GINConv(hidden_size))
            elif gnn_name == 'gcn':
                if algorithm_aligned == 'bf':
                    self.convs.append(GCNConv_BF(hidden_size))
                elif algorithm_aligned == 'mst':
                    self.convs.append(GCNConv_GRIN(hidden_size))
                else:
                    self.convs.append(GCNConv(hidden_size))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_name))
            
            if norm_layer.split('_')[0] == 'batch':
                if norm_layer.split('_')[-1] == 'notrack':
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_size, track_running_stats=False, affine=False))
                else:
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_size))
            elif norm_layer.split('_')[0] == 'instance':
                self.batch_norms.append(InstanceNorm(hidden_size))
            elif norm_layer.split('_')[0] == 'layer':
                self.batch_norms.append(LayerNorm(hidden_size))
            elif norm_layer.split('_')[0] == 'graph':
                self.batch_norms.append(GraphNorm(hidden_size))
            elif norm_layer.split('_')[0] == 'size':
                self.batch_norms.append(GraphSizeNorm())
            elif norm_layer.split('_')[0] == 'pair':
                self.batch_norms.append(PairNorm(hidden_size))
            elif norm_layer.split('_')[0] == 'group':
                self.batch_norms.append(DiffGroupNorm(hidden_size, groups=4))
            else:
                raise ValueError('Undefined normalization layer called {}'.format(norm_layer))
        if norm_layer.split('_')[1] == 'size':
            self.graph_size_norm = GraphSizeNorm()
        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden_size, 2*hidden_size), torch.nn.BatchNorm1d(2*hidden_size), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*hidden_size, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU()))


    def forward(self, *args):
        if len(args) == 1:
            # Case 1: batched_data input
            batched_data = args[0]
            x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        elif len(args) == 4:
            # Case 2: separate inputs
            x, edge_index, edge_attr, batch = args
        else:
            raise ValueError("forward expects either 1 batched_data argument or 4 separate arguments (x, edge_index, edge_attr, batch)")
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        if self.encode_atom:
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]
        if self.algorithm_aligned == 'mst':
            visited = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            ### Message passing among graph nodes
            if self.algorithm_aligned == 'mst':
                h, visited = self.convs[layer](h_list[layer], edge_index, edge_attr, visited)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            if self.norm_layer.split('_')[0] == 'batch':
                h = self.batch_norms[layer](h)
            else:
                h = self.batch_norms[layer](h, batch)
            if self.norm_layer.split('_')[1] == 'size':
                h = self.graph_size_norm(h, batch)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.relu(h)
                h = F.dropout(h, self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation, h_list