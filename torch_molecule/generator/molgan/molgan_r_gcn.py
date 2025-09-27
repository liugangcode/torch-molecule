import torch
import torch.nn as nn

class RelationalGCNLayer(nn.Module):
    """
    Relational Graph Convolutional Layer for fully connected dense graphs.
    Input:
        - node_feats: [batch, num_nodes, in_dim]
        - adj: [batch, num_relations, num_nodes, num_nodes]
    Output:
        - node_feats: [batch, num_nodes, out_dim]
    """
    def __init__(self, in_dim, out_dim, num_relations, use_bias=True):
        super(RelationalGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # One weight matrix per relation/bond type
        self.rel_weights = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rel_weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, node_feats, adj):
        # node_feats: [batch, num_nodes, in_dim]
        # adj: [batch, num_relations, num_nodes, num_nodes]
        batch_size, num_nodes, _ = node_feats.shape

        out = torch.zeros(batch_size, num_nodes, self.out_dim, device=node_feats.device)

        for rel in range(self.num_relations):
            # Multiply node features by relation weight
            # [batch, num_nodes, in_dim] @ [in_dim, out_dim] -> [batch, num_nodes, out_dim]
            h_rel = torch.matmul(node_feats, self.rel_weights[rel])
            # Propagate messages using adjacency for this relation:
            # [batch, num_nodes, out_dim] ← [batch, num_nodes, num_nodes] @ [batch, num_nodes, out_dim]
            # Here adj[:, rel, :, :] gives [batch, num_nodes, num_nodes]
            out += torch.bmm(adj[:, rel], h_rel)

        if self.bias is not None:
            out += self.bias

        return out  # You can add activation after this (ReLU, LeakyReLU, etc.)

