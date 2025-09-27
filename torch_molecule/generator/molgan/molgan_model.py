from typing import Optional
import torch
from dataclasses import dataclass
from .molgan_gen_disc import *
from .molgan_dataset import MolGANDataset


class MolGANModel(torch.nn.Module):
    def __init__(
        self,
        generator_config: MolGANGeneratorConfig,
        discriminator_config: MolGANDiscriminatorConfig,
        reward_network_config: Optional[MolGANDiscriminatorConfig] = None,
    ):
        super(MolGANModel, self).__init__()

        # Initialize generator and discriminator
        self.gen = MolGANGenerator(generator_config)
        self.disc = MolGANDiscriminator(discriminator_config)

        # By default, the reward network is the same as the discriminator
        self.reward_net = (
            MolGANDiscriminator(reward_network_config)
            if reward_network_config is not None
            else MolGANDiscriminator(discriminator_config)
        )
