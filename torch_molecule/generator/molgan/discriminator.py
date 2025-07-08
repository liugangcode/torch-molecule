import torch.nn as nn
from .gan_utils import RelationalGCNLayer


class MolGANDiscriminatorConfig:
    """
    Configuration class for MolGAN Discriminator.

    Stores architectural hyperparameters and allows modular configuration.
    """

    def __init__(self,
                 num_atom_types=5,
                 num_bond_types=4,
                 num_nodes=9,
                 hidden_dim=128,
                 num_layers=2):
        """
        Parameters
        ----------
        num_atom_types : int
            Number of atom types in node features (input channels).

        num_bond_types : int
            Number of bond types (number of relational edge types).

        num_nodes : int
            Max number of nodes in the graph (used for flattening before readout).

        hidden_dim : int
            Hidden dimension size for R-GCN layers.

        num_layers : int
            Number of stacked R-GCN layers.
        """
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers




class MolGANDiscriminator(nn.Module):
    """
    Discriminator network for MolGAN using stacked Relational GCNs.
    """

    def __init__(self, config: MolGANDiscriminatorConfig):
        super().__init__()
        self.config = config

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(
            RelationalGCNLayer(config.num_atom_types, config.hidden_dim, config.num_bond_types)
        )

        for _ in range(1, config.num_layers):
            self.gcn_layers.append(
                RelationalGCNLayer(config.hidden_dim, config.hidden_dim, config.num_bond_types)
            )

        self.readout = nn.Sequential(
            nn.Linear(config.num_nodes * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, adj, node):
        """
        Parameters:
        adj: Tensor of shape [B, Y, N, N] -- adjacency tensor
        node: Tensor of shape [B, N, T]   -- one-hot or softmax node features

        Returns:
        Tensor of shape [B] with real/fake logits
        """
        h = node
        for gcn in self.gcn_layers:
            h = gcn(adj, h)

        h = h.view(h.size(0), -1)
        return self.readout(h).squeeze(-1)







