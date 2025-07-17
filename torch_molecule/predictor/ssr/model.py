import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import copy
# from torch_scatter import scatter
from .utils import scatter

from ...nn import GNN_node, GNN_node_Virtualnode, MLP
from ...utils import init_weights

class SSR(nn.Module):
    def __init__(
        self,
        num_task,
        num_layer,
        coarse_pool='mean',
        hidden_size=300,
        gnn_type="gin-virtual",
        drop_ratio=0.5,
        norm_layer="batch_norm",
        graph_pooling="mean",
        augmented_feature=['maccs', 'morgan']
    ):

        super(SSR, self).__init__()
        gnn_name = gnn_type.split("-")[0]
        self.num_task = num_task
        self.hidden_size = hidden_size
        self.coarse_pool = coarse_pool

        if "virtual" in gnn_type:
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
            self.graph_encoder = GNN_node(
                num_layer,
                hidden_size,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError(f"Invalid graph pooling type {graph_pooling}.")

        graph_dim = hidden_size
        self.augmented_feature = augmented_feature
        if augmented_feature:
            if "morgan" in augmented_feature:
                graph_dim += 1024
            if "maccs" in augmented_feature:
                graph_dim += 167
        self.predictor = MLP(graph_dim, hidden_features=2 * hidden_size, out_features=num_task)
    
    def l2diff(self, x1, x2):
        """Compute L2 difference between tensors"""
        return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))
        
    def moment_diff(self, sx1, sx2, k, og_batch, coarse_batch):
        """Compute difference between k-th moments"""
        ss1 = scatter(sx1**k, og_batch, dim=0, reduce='mean')
        ss2 = scatter(sx2**k, coarse_batch, dim=0, reduce='mean')
        return self.l2diff(ss1, ss2)
        
    def cmd(self, x1, x2, og_batch, coarse_batch,n_moments):
        """Central Moment Discrepancy between x1 and x2 distributions"""
        # Compute means
        mx1 = scatter(x1, og_batch, dim=0, reduce='mean')
        mx2 = scatter(x2, coarse_batch, dim=0, reduce='mean')
        # Center features
        sx1 = x1 - mx1[og_batch]
        sx2 = x2 - mx2[coarse_batch]
        
        # First moment difference
        dm = self.l2diff(mx1, mx2)
        scms = dm
        
        # Higher order moments
        for i in range(n_moments-1):
            scms = scms + self.moment_diff(sx1, sx2, i+2, og_batch, coarse_batch)
            
        return scms.mean()
            
    def pool_batch_by_graph(self, clusters, original_batch):
        """
        Build a new batch vector for coarse nodes so that all coarse nodes derived from the same
        original graph get the same graph label. Here, we iterate graph-by-graph.
        
        Args:
            clusters (Tensor): Cluster assignments for each original node, shape [num_original_nodes].
            original_batch (Tensor): Original batch vector (graph assignments) for each node.
        
        Returns:
            Tensor: New batch vector for coarse nodes, of shape [num_coarse_nodes].
        """
        unique_graphs = torch.unique(original_batch)
        new_batch_list = []
        for g in unique_graphs:
            # Get indices for nodes belonging to graph g.
            mask = (original_batch == g)
            # For these nodes, get their cluster ids.
            clusters_in_g = clusters[mask]
            # Determine the unique clusters in this graph.
            unique_clusters_in_g = torch.unique(clusters_in_g)
            # For each coarse node from graph g, assign the same graph id (g.item())
            new_batch_list.extend([g.item()] * unique_clusters_in_g.numel())
        return torch.tensor(new_batch_list, device=original_batch.device, dtype=torch.long)


    def prepare_coarsened_batch(self, batch, ratio):
        """
        Create a coarsened version of batch for forward pass with edge attributes,
        ensuring that the new batch assignment preserves the original graph labels.
        """
        new_batch = copy.deepcopy(batch)
        coarse_ratio_postfix = str(int(ratio * 100))
        
        # 1. Get clusters (one per original node) and ensure long type.
        clusters = getattr(batch, f"clusters_{coarse_ratio_postfix}").long()  # shape: [num_nodes]
        
        # Compute a factor to disambiguate clusters from different graphs.
        factor = clusters.max().item() + 1  # All clusters are in [0, factor-1]
        # Create a global key that combines the original batch assignment and clusters.
        global_coarse_key = batch.batch * factor + clusters  # shape: [num_nodes]
        
        # Get the unique global keys and the inverse mapping.
        unique_global_keys, inverse_indices = torch.unique(global_coarse_key, return_inverse=True)
        num_coarse_nodes = unique_global_keys.numel()
        
        # 2. Pool node features using the global coarse key.
        new_x = scatter(batch.x, inverse_indices, dim=0, reduce=self.coarse_pool, dim_size=num_coarse_nodes)
        new_batch.x = new_x

        # 3. Build the new batch vector for coarse nodes:
        # The graph assignment for each coarse node is simply:
        #   graph_id = unique_global_key // factor
        new_batch_vec = unique_global_keys // factor
        new_batch.batch = new_batch_vec
        
        # 4. Process the coarsened edge index.
        # Assume that the stored coarsened edge index is based on original node indices.
        # Map each node index to its global coarse key.
        coarse_edge_index = getattr(batch, f"coarsened_edge_index_{coarse_ratio_postfix}")
        # For each index in coarse_edge_index, replace with its global coarse key.
        temp_edge_index = torch.stack([
            global_coarse_key[coarse_edge_index[0]],
            global_coarse_key[coarse_edge_index[1]]
        ])
        # Build a mapping from global key to new index from unique_global_keys.
        mapping = {old.item(): new for new, old in enumerate(unique_global_keys)}
        new_edge_index = torch.zeros_like(temp_edge_index)
        for i in range(temp_edge_index.size(1)):
            new_edge_index[0, i] = mapping[temp_edge_index[0, i].item()]
            new_edge_index[1, i] = mapping[temp_edge_index[1, i].item()]
        new_batch.edge_index = new_edge_index
        
        # 5. Process edge attributes, if available.
        if hasattr(batch, f"coarsened_edge_attr_{coarse_ratio_postfix}"):
            edge_attr = getattr(batch, f"coarsened_edge_attr_{coarse_ratio_postfix}")
            new_batch.edge_attr = torch.round(edge_attr).long()
        
        new_batch.num_nodes = num_coarse_nodes
        '''
        # Debug prints:
        print("\n=== Feature and Cluster Info ===")
        print(f"Original x shape: {batch.x.shape}")
        print(f"Original batch shape: {batch.batch.shape}")
        print(f"Number of unique clusters (globally disambiguated): {num_coarse_nodes}")
        print("\n=== Final Dimensions ===")
        print(f"New x shape: {new_batch.x.shape}")      # Should be [num_coarse_nodes, feature_dim]
        print(f"New batch shape: {new_batch.batch.shape}")  # Should be [num_coarse_nodes]
        print(f"New edge_index shape: {new_batch.edge_index.shape}")
        '''
        # Validate dimensions.
        assert new_batch.batch.numel() == num_coarse_nodes, \
            f"Batch size {new_batch.batch.numel()} doesn't match number of coarse nodes {num_coarse_nodes}"
        assert new_batch.x.shape[0] == num_coarse_nodes, \
            f"Number of node features {new_batch.x.shape[0]} doesn't match number of coarse nodes {num_coarse_nodes}"
        
        return new_batch

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

    def _augmented_graph_features(self, batched_data, h_rep):
        if self.augmented_feature:
            if 'morgan' in self.augmented_feature:
                morgan = batched_data.morgan.type_as(h_rep)
                h_rep = torch.cat((h_rep, morgan), dim=1)
            if 'maccs' in self.augmented_feature:
                maccs = batched_data.maccs.type_as(h_rep)
                h_rep = torch.cat((h_rep, maccs), dim=1)
        return h_rep


    def compute_loss(self, batched_data, criterion, coarse_ratios=[0.8, 0.9], cmd_coeff=0.1, fine_grained=True, n_moments=5):
        """Compute loss with SSR regularization"""
        # Original forward pass
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        h_rep = self._augmented_graph_features(batched_data, h_rep)
        prediction = self.predictor(h_rep)
        
        # Compute prediction loss
        target = batched_data.y.to(torch.float32)
        is_labeled = batched_data.y == batched_data.y  # Check for NaN
        pred_loss = criterion(prediction.to(torch.float32)[is_labeled], target[is_labeled]).mean()
        
        # Initialize SSR loss
        ssr_loss = torch.tensor(0.0, device=batched_data.x.device)
        
        # Process each coarsening ratio for SSR
        for ratio in coarse_ratios:
            # Skip if coarsened data wasn't pre-computed
            if not hasattr(batched_data, f"coarsened_edge_index_{str(int(ratio*100))}"):
                continue
                
            # Prepare coarsened batch
            coarse_batch = self.prepare_coarsened_batch(batched_data, ratio)
            
            # Get node embeddings for coarsened graph
            coarse_h_node, _ = self.graph_encoder(coarse_batch)
            
            # Compute CMD loss
            if fine_grained:
                # Node-level CMD
                ssr_loss = ssr_loss + self.cmd(h_node, 
                                              coarse_h_node,
                                              batched_data.batch,
                                              coarse_batch.batch,n_moments)
            else:
                # Graph-level CMD
                coarse_h_rep = self.pool(coarse_h_node, coarse_batch.batch)
                ssr_loss = ssr_loss + torch.norm(h_rep - coarse_h_rep, dim=1).mean()
        
        # Compute total loss
        ssr_loss = cmd_coeff * ssr_loss
        if ssr_loss > 10:
            import warnings
            warnings.warn(f"SSR loss is too large: {ssr_loss}, truncating to 10")
            ssr_loss = 10
        total_loss = pred_loss + ssr_loss
        
        return total_loss, pred_loss, ssr_loss
        
    def forward(self, batched_data):
        """Standard forward pass (for inference without SSR)"""
        h_node, _ = self.graph_encoder(batched_data)
        h_rep = self.pool(h_node, batched_data.batch)
        h_rep = self._augmented_graph_features(batched_data, h_rep)
        prediction = self.predictor(h_rep)
        return {"prediction": prediction}