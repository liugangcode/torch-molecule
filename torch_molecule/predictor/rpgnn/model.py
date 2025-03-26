import torch
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from ...nn import GNN_node, GNN_node_Virtualnode, MLP
from ...utils import init_weights


class RPGNN(nn.Module):
    def __init__(
        self,
        num_task,
        num_layer,
        num_node_feature,
        fixed_size=32,
        num_perm=3,
        hidden_size=300,
        gnn_type="gin-virtual",
        drop_ratio=0.5,
        norm_layer="batch_norm",
        graph_pooling="mean",
        augmented_feature=['maccs', 'morgan']
    ):
        """
        Random Permutation Graph Neural Network (RPGNN)
        
        Args:
            num_task: Number of prediction tasks
            num_layer: Number of GNN layers
            num_node_feature: Number of node features
            fixed_size: Size of position encoding vectors
            num_perm: Number of random permutations to average over
            hidden_size: Hidden dimension size
            gnn_type: Type of GNN to use (gin, gin-virtual, gcn, gcn-virtual)
            drop_ratio: Dropout ratio
            norm_layer: Type of normalization layer
            graph_pooling: Type of graph pooling (mean, sum, max)
            augmented_feature: List of additional molecular features to use
        """
        super(RPGNN, self).__init__()
        
        gnn_name = gnn_type.split("-")[0]
        self.num_task = num_task
        self.hidden_size = hidden_size
        self.fixed_size = fixed_size
        self.num_perm = num_perm
        self.augmented_feature = augmented_feature
        self.num_node_feature = num_node_feature        
        # Register fixed-size node IDs (position encoding)
        self.register_buffer("node_ids", torch.eye(self.fixed_size))
        
        # GNN selection
        if "virtual" in gnn_type:
            self.gnn = GNN_node_Virtualnode(
                num_layer=num_layer,
                hidden_size=hidden_size,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer
            )
        else:
            self.gnn = GNN_node(
                num_layer=num_layer,
                hidden_size=hidden_size,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer
            )
        
        # Graph pooling selection
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Invalid graph pooling type: {graph_pooling}")
            
        # Calculate final prediction dimension
        graph_dim = hidden_size+self.fixed_size
        if augmented_feature:
            if "morgan" in augmented_feature:
                graph_dim += 1024
            if "maccs" in augmented_feature:
                graph_dim += 167
                
        # Final predictor MLP
        self.predictor = MLP(
            in_features=graph_dim,
            hidden_features=2 * hidden_size,
            out_features=num_task
        )

    def initialize_parameters(self, seed=None):
        """Initialize all model parameters."""
        if seed is not None:
            torch.manual_seed(seed)
            
        # Initialize GNN and predictor
        init_weights(self.gnn)
        init_weights(self.predictor)
        
        # Apply reset_parameters to all modules that have it
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        
        self.apply(reset_parameters)

    def _augmented_graph_features(self, batched_data, h_rep):
        """Add augmented molecular features to the graph representation."""
        if not self.augmented_feature:
            return h_rep
            
        if "morgan" in self.augmented_feature:
            morgan = batched_data.morgan.type_as(h_rep)
            h_rep = torch.cat((h_rep, morgan), dim=1)
        if "maccs" in self.augmented_feature:
            maccs = batched_data.maccs.type_as(h_rep)
            h_rep = torch.cat((h_rep, maccs), dim=1)
            
        return h_rep

    def forward(self, batched_data):
        """Forward pass of RPGNN."""
        # Extract original data
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        
        # Initialize final output
        out = None
        
        # Average over multiple permutations
        for _ in range(self.num_perm):
            temp_data = batched_data.clone()
            # Create new_x with same dtype as x
            new_x = torch.empty(x.size(0), x.size(1) + self.fixed_size, dtype=x.dtype, device=x.device)
            
            for graph_idx in range(batch.max().item() + 1):
                node_indices = (batch == graph_idx).nonzero().squeeze(1)
                graph_size = node_indices.size(0)
                perm = torch.randperm(graph_size)

                node_ids = self.node_ids.repeat(
                    graph_size // self.fixed_size + 1, 1
                )[:graph_size]
                permuted_node_ids = node_ids[perm, :]
                
                # Convert both tensors to the same type as x before concatenation
                new_x[node_indices] = torch.cat([x[node_indices].to(x.dtype), permuted_node_ids.to(x.dtype)], dim=1)
            temp_data.x = new_x[:,:self.num_node_feature]
            # Process through GNN
            h_v, _ = self.gnn(temp_data)

            fused_h_v = torch.cat([h_v, new_x[:,self.num_node_feature:]], dim=1)

            # Pool node representations to graph representations
            h_rep = self.pool(fused_h_v, temp_data.batch)
            
            # Add augmented features
            h_rep = self._augmented_graph_features(temp_data, h_rep)
            
            # Accumulate output
            if out is None:
                out = h_rep / self.num_perm
            else:
                out += h_rep / self.num_perm
        
        # Final prediction
        prediction = self.predictor(out)
        
        return {"prediction": prediction}

    def compute_loss(self, batched_data, criterion):
        out = self.forward(batched_data)
        prediction = out['prediction']
        target = batched_data.y.to(torch.float32)
        is_labeled = batched_data.y == batched_data.y
        loss = criterion(prediction.to(torch.float32)[is_labeled], target[is_labeled]).mean()
        
        return loss