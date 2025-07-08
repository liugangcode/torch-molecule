import torch
import torch.nn as nn
import torch.nn.functional as F



class MolGANConfig:
    """
    Configuration class for MolGAN Generator and Discriminator.

    This class stores hyperparameters and architectural details used to construct
    the MolGAN generator and other related modules. It allows modular control over
    model depth, input/output dimensionality, and Gumbel-softmax behavior.
    """
    def __init__(self,
                 latent_dim=56,
                 hidden_dims=[128, 128, 256],
                 num_nodes=9,
                 num_atom_types=5,
                 num_bond_types=4,
                 tau=1.0):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_nodes = num_nodes
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.tau = tau



class MolGANGenerator(nn.Module):

    """
    Generator network for MolGAN.

    Maps a latent vector z to a molecular graph represented by:
    - Adjacency tensor A ∈ [B, Y, N, N] (bonds)
    - Node features X ∈ [B, N, T] (atoms)

    Uses Gumbel-Softmax to approximate discrete molecular structure.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        output_dim = (config.num_nodes * config.num_atom_types) + \
                     (config.num_nodes * config.num_nodes * config.num_bond_types)

        layers = []
        input_dim = config.latent_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))

        self.fc = nn.Sequential(*layers)

    def forward(self, z):
        B = z.size(0)
        out = self.fc(z)

        N, T, Y = self.config.num_nodes, self.config.num_atom_types, self.config.num_bond_types
        node_size = N * T
        adj_size = N * N * Y

        node_flat, adj_flat = torch.split(out, [node_size, adj_size], dim=1)
        node = node_flat.view(B, N, T)
        adj = adj_flat.view(B, Y, N, N)

        # Gumbel-softmax
        node = F.gumbel_softmax(node, tau=self.config.tau, hard=True, dim=-1)
        adj = F.gumbel_softmax(adj, tau=self.config.tau, hard=True, dim=1)

        return adj, node
