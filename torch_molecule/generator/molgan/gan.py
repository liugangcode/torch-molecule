import torch
import torch.nn as nn
from .generator import MolGANGenerator
from .discriminator import MolGANDiscriminator

class MolGAN(nn.Module):
    """
    Combined MolGAN model: generator + discriminator
    """

    def __init__(self, gen_config, disc_config):
        super().__init__()
        self.generator = MolGANGenerator(gen_config)
        self.discriminator = MolGANDiscriminator(disc_config)

    def generate(self, z):
        """Forward pass through generator only."""
        return self.generator(z)

    def discriminate(self, adj, node):
        """Forward pass through discriminator only."""
        return self.discriminator(adj, node)

    def forward(self, z):
        """
        Combined forward pass (generator → discriminator).
        Used for adversarial training.
        """
        adj_fake, node_fake = self.generator(z)
        pred_fake = self.discriminator(adj_fake, node_fake)
        return adj_fake, node_fake, pred_fake
