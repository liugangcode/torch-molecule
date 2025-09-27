import torch
from dataclasses import dataclass
from typing import Tuple
from .molgan_r_gcn import RelationalGCNLayer  # Local import to avoid circular dependency

@dataclass
class MolGANGeneratorConfig:
    def __init__(
        self,
        z_dim: int = 32,
        g_conv_dim: int = 64,
        d_conv_dim: int = 64,
        g_num_layers: int = 3,
        d_num_layers: int = 3,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        max_num_atoms: int = 9,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ):
        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.d_conv_dim = d_conv_dim
        self.g_num_layers = g_num_layers
        self.d_num_layers = d_num_layers
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.max_num_atoms = max_num_atoms
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm


# MolGAN Generotor
class MolGANGenerator(torch.nn.Module):
    def __init__(self, config: MolGANGeneratorConfig):
        super(MolGANGenerator, self).__init__()
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.g_num_layers = config.g_num_layers
        self.num_atom_types = config.num_atom_types
        self.num_bond_types = config.num_bond_types
        self.max_num_atoms = config.max_num_atoms
        self.dropout = config.dropout
        self.use_batchnorm = config.use_batchnorm

        layers = []
        input_dim = self.z_dim
        for i in range(self.g_num_layers):
            output_dim = self.g_conv_dim * (2 ** i)
            layers.append(torch.nn.Linear(input_dim, output_dim))
            if self.use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(output_dim))
            layers.append(torch.nn.ReLU())
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))
            input_dim = output_dim

        self.fc_layers = torch.nn.Sequential(*layers)
        self.atom_fc = torch.nn.Linear(input_dim, self.max_num_atoms * self.num_atom_types)
        self.bond_fc = torch.nn.Linear(input_dim, self.num_bond_types * self.max_num_atoms * self.max_num_atoms)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = z.size(0)
        h = self.fc_layers(z)
        atom_logits = self.atom_fc(h).view(batch_size, self.max_num_atoms, self.num_atom_types)
        # Output bond logits with [batch, num_bond_types, max_num_atoms, max_num_atoms] order
        bond_logits = self.bond_fc(h).view(batch_size, self.num_bond_types, self.max_num_atoms, self.max_num_atoms)
        return atom_logits, bond_logits



# MolGAN Discriminator
@dataclass
class MolGANDiscriminatorConfig:
    def __init__(
        self,
        in_dim: int = 5,             # Number of atom types (node feature dim). Typically set automatically.
        hidden_dim: int = 64,        # Hidden feature/channel size for GCN layers.
        num_layers: int = 3,         # Number of R-GCN layers (depth).
        num_relations: int = 4,      # Number of bond types (relation types per edge).
        max_num_atoms: int = 9,      # Max node count in padded tensor.
        dropout: float = 0.0,        # Dropout between layers.
        use_batchnorm: bool = True,  # BatchNorm or similar normalization.
        readout: str = 'sum',        # Readout type (sum/mean/max for pooling nodes to graph-level vector)
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.max_num_atoms = max_num_atoms
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.readout = readout


class MolGANDiscriminator(torch.nn.Module):
    def __init__(self, config: MolGANDiscriminatorConfig):
        super(MolGANDiscriminator, self).__init__()

        self.in_dim = config.in_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_relations = config.num_relations
        self.max_num_atoms = config.max_num_atoms
        self.dropout = config.dropout
        self.use_batchnorm = config.use_batchnorm
        self.readout = config.readout

        layers = []
        input_dim = self.in_dim
        for i in range(self.num_layers):
            output_dim = self.hidden_dim * (2 ** i)
            layers.append(RelationalGCNLayer(input_dim, output_dim, self.num_relations))
            if self.use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(self.max_num_atoms))
            layers.append(torch.nn.LeakyReLU(0.2))
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))
            input_dim = output_dim

        self.gcn_layers = torch.nn.ModuleList(layers)
        self.fc = torch.nn.Linear(input_dim, 1)

    def forward(
        self,
        atom_feats: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        # atom_feats: [batch, max_num_atoms, num_atom_types]
        # adj: [batch, num_bond_types, max_num_atoms, max_num_atoms]
        # mask: [batch, max_num_atoms] (float, 1=real, 0=pad)
        h = atom_feats
        for layer in self.gcn_layers:
            if isinstance(layer, RelationalGCNLayer):
                h = layer(h, adj)
            else:
                # If using BatchNorm1d, input should be [batch, features, nodes]
                if isinstance(layer, torch.nn.BatchNorm1d):
                    # Permute for batchnorm: [batch, nodes, features] → [batch, features, nodes]
                    h = layer(h.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    h = layer(h)

        # MASKED GRAPH READOUT
        # mask: [batch, max_num_atoms] float
        mask = mask.unsqueeze(-1)  # [batch, max_num_atoms, 1]
        h_masked = h * mask  # zeros padded nodes

        if self.readout == 'sum':
            g = h_masked.sum(dim=1)   # [batch, hidden_dim]
        elif self.readout == 'mean':
            # Prevent divide-by-zero with (mask.sum(dim=1, keepdim=True)+1e-8)
            g = h_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        elif self.readout == 'max':
            # Set padded to large neg, then max
            h_masked_pad = h.clone()
            h_masked_pad[mask.squeeze(-1) == 0] = float('-inf')
            g, _ = h_masked_pad.max(dim=1)
        else:
            raise ValueError(f"Unknown readout type: {self.readout}")

        out = self.fc(g)  # [batch, 1]
        return out.squeeze(-1)  # [batch]
