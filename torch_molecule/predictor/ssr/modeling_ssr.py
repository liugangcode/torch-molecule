import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Literal, Type
from dataclasses import dataclass, field

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from .model import SSR
from ...utils import graph_from_smiles
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

@dataclass
class SSRMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements a SizeShiftReg model with the GNN.
    
    References
    ----------
    - Paper: SizeShiftReg: a Regularization Method for Improving Size-Generalization in Graph Neural Networks.
      https://arxiv.org/abs/2207.07888
    - Reference Code: https://github.com/DavideBuffelli/SizeShiftReg/tree/main
    
    Parameters
    ----------
    coarse_ratios : List[float], default=[0.8, 0.9]
        List of ratios for graph coarsening. Each ratio determines the percentage of 
        nodes to keep in the coarsened graph.
    cmd_coeff : float, default=0.1
        Weight for CMD (Central Moment Discrepancy) loss. Controls the strength of 
        the size-shift regularization.
    fine_grained : bool, default=True
        Whether to use fine-grained CMD. When True, matches distributions at a more 
        detailed level.
    n_moments : int, default=5
        Number of moments to match in the CMD calculation. Higher values capture 
        more complex distribution characteristics.
    """
    # SSR-specific parameters
    coarse_ratios: List[float] = field(default_factory=lambda: [0.8, 0.9])
    cmd_coeff: float = field(default=0.1)
    fine_grained: bool = field(default=True)
    n_moments: int = field(default=5)
    coarse_pool: str = field(default='mean')

    # Other Non-init fields
    model_name: str = "SSRMolecularPredictor"
    model_class: Type[SSR] = field(default=SSR, init=False)

    def __post_init__(self):
        super().__post_init__()

    @staticmethod
    def _get_param_names() -> List[str]:
        return GNNMolecularPredictor._get_param_names() + [
            "coarse_ratios",
            "cmd_coeff",
            "fine_grained",
            "n_moments",
            "coarse_pool",
        ]

    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["cmd_coeff"] = ParameterSpec(ParameterType.FLOAT, (0.01, 1.0))
        search_space["n_moments"] = ParameterSpec(ParameterType.INTEGER, (1, 10))
        return search_space

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        base_params = super()._get_model_params(checkpoint)
        return base_params


    def _convert_to_pytorch_data(self, X, y=None):
        """Convert SMILES to PyTorch Geometric data with coarsened versions, preserving edge attributes."""
        if self.verbose:
            iterator = tqdm(enumerate(X), desc="Converting molecules to graphs", total=len(X))
        else:
            iterator = enumerate(X)
            
        pyg_graph_list = []
        for idx, smiles_or_mol in iterator:
            if y is not None:
                properties = y[idx]
            else:
                properties = None
                
            # Convert SMILES to graph
            graph = graph_from_smiles(smiles_or_mol, properties, self.augmented_feature)
            g = Data()
            g.num_nodes = graph["num_nodes"]
            g.edge_index = torch.from_numpy(graph["edge_index"])

            # Standard attributes
            if graph["edge_feat"] is not None:
                g.edge_attr = torch.from_numpy(graph["edge_feat"])
            
            if graph["node_feat"] is not None:
                g.x = torch.from_numpy(graph["node_feat"])

            if graph["y"] is not None:
                g.y = torch.from_numpy(graph["y"])
                
            if graph.get("morgan") is not None:
                g.morgan = torch.tensor(graph["morgan"], dtype=torch.int8).view(1, -1)
                
            if graph.get("maccs") is not None:
                g.maccs = torch.tensor(graph["maccs"], dtype=torch.int8).view(1, -1)

            # Add coarsened versions
            for ratio in self.coarse_ratios:
                num_clusters = max(1, int(g.num_nodes * ratio))
                coarse_ratio_postfix = str(int(ratio*100))
                
                # Get coarsened graph with edge attributes
                coarse_edge_index, coarse_edge_attr, clusters = self.spectral_graph_coarsening(g, num_clusters)
                
                # Add attributes to graph
                setattr(g, f"coarsened_edge_index_{coarse_ratio_postfix}", coarse_edge_index)
                if hasattr(g, 'edge_attr'):
                    setattr(g, f"coarsened_edge_attr_{coarse_ratio_postfix}", coarse_edge_attr)
                setattr(g, f"num_coarse_nodes_{coarse_ratio_postfix}", torch.tensor(num_clusters))
                setattr(g, f"clusters_{coarse_ratio_postfix}", clusters)
                
            pyg_graph_list.append(g)
            
        return pyg_graph_list

    def spectral_graph_coarsening(self, graph, num_clusters):
        """Coarsen graph based on spectral clustering while preserving edge attributes"""
        # Extract node features and edge information
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
        num_nodes = graph.num_nodes
        edge_attr = edge_attr.float()

        # Convert to adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # Calculate Laplacian
        degree = adj.sum(dim=1)
        degree_mat = torch.diag(degree)
        laplacian = degree_mat - adj
        
        # Compute eigenvectors
        if num_clusters < num_nodes - 1:
            eigvals, eigvecs = torch.linalg.eigh(laplacian)
            # Use smallest non-zero eigenvalues
            indices = torch.argsort(eigvals)[1:num_clusters+1]  # Skip first eigenvector
            fiedler_vectors = eigvecs[:, indices]
            
            # Use k-means for clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(fiedler_vectors.cpu().numpy())
            clusters = torch.tensor(clusters, device=edge_index.device)
        else:
            clusters = torch.arange(num_nodes, device=edge_index.device)
        
        # Create coarsened edge index and attributes
        coarse_edge_index = []
        coarse_edge_attr = []
        
        # Store mapping from (cluster_i, cluster_j) to list of edge indices
        cluster_edges = {}
        
        # Group edges by their clusters
        for e_idx in range(edge_index.shape[1]):
            src, dst = edge_index[0, e_idx], edge_index[1, e_idx]
            c_src, c_dst = clusters[src], clusters[dst]
            
            # Skip self-loops in coarsened graph
            if c_src == c_dst:
                continue
                
            # Sort clusters to avoid duplicates
            if c_src > c_dst:
                c_src, c_dst = c_dst, c_src
                
            key = (c_src.item(), c_dst.item())
            if key not in cluster_edges:
                cluster_edges[key] = []
            
            if edge_attr is not None:
                cluster_edges[key].append(edge_attr[e_idx])
        
        # Create coarsened edges
        for (c_src, c_dst), edge_attrs in cluster_edges.items():
            coarse_edge_index.append([c_src, c_dst])
            coarse_edge_index.append([c_dst, c_src])  # Add both directions
            
            if edge_attr is not None:
                # Aggregate edge attributes using mean
                mean_attr = torch.stack(edge_attrs).mean(dim=0)
                coarse_edge_attr.append(mean_attr)
                coarse_edge_attr.append(mean_attr)  # Same for both directions
        
        # Convert to tensors
        if coarse_edge_index:
            coarse_edge_index = torch.tensor(coarse_edge_index, dtype=torch.long, 
                                            device=edge_index.device).t()
            if edge_attr is not None:
                coarse_edge_attr = torch.stack(coarse_edge_attr)
        else:
            coarse_edge_index = torch.zeros((2, 0), dtype=torch.long, 
                                        device=edge_index.device)
            if edge_attr is not None:
                coarse_edge_attr = torch.zeros((0, edge_attr.size(1)), 
                                            device=edge_attr.device)
        
        return coarse_edge_index, coarse_edge_attr, clusters

    def _train_epoch(self, train_loader, optimizer, epoch):
        """Training logic for one epoch with SSR."""
        self.model.train()
        losses = []
        pred_losses = []
        ssr_losses = []

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )

        for batch in iterator:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            # Forward pass and loss computation
            total_loss, pred_loss, ssr_loss = self.model.compute_loss(batch, self.loss_criterion, self.coarse_ratios, self.cmd_coeff, self.fine_grained, self.n_moments)

            # Backward pass
            total_loss.backward()

            # Compute gradient norm if gradient clipping is enabled
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            optimizer.step()

            losses.append(total_loss.item())
            pred_losses.append(pred_loss.item())
            ssr_losses.append(ssr_loss.item())

            # Update progress bar if using tqdm
            if self.verbose:
                iterator.set_postfix({
                    "Epoch": epoch,
                    "Total loss": f"{total_loss.item():.4f}",
                    "Pred loss": f"{pred_loss.item():.4f}",
                    "ssr_loss": f"{ssr_loss.item():.4f}"
                })

        # Return all loss components for logging
        return losses