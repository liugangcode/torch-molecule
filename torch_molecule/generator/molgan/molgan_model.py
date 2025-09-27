from typing import Optional
import torch
import torch.nn as nn
from .molgan_gen_disc import *


class MolGANModel(nn.Module):
    def __init__(
        self,
        generator_config: Optional[MolGANGeneratorConfig] = None,
        discriminator_config: Optional[MolGANDiscriminatorConfig] = None,
        reward_network_config: Optional[MolGANDiscriminatorConfig] = None,
    ):
        super(MolGANModel, self).__init__()

        # Initialize generator and discriminator
        self.gen_config = generator_config if generator_config is not None else MolGANGeneratorConfig()
        self.gen = MolGANGenerator(self.gen_config)

        self.disc_config = discriminator_config if discriminator_config is not None else MolGANDiscriminatorConfig()
        self.disc = MolGANDiscriminator(self.disc_config)

        # By default, the reward network is the same as the discriminator
        self.reward_net = (
            MolGANDiscriminator(reward_network_config)
            if reward_network_config is not None
            else MolGANDiscriminator(self.disc_config)
        )


    def generate(self, batch_size: int):
        """Generate a batch of molecules."""
        return self.gen(batch_size)


    def discriminate(
        self,
        atom_type_matrix: torch.Tensor,
        bond_type_tensor: torch.Tensor,
        molecule_mask: Optional[torch.Tensor],
    ):
        """Discriminate a batch of molecules."""
        return self.disc(atom_type_matrix, bond_type_tensor, molecule_mask)

    def reward(
        self,
        atom_type_matrix: torch.Tensor,
        bond_type_tensor: torch.Tensor,
        molecule_mask: Optional[torch.Tensor],
    ):
        """Compute reward for a batch of molecules."""
        return self.reward_net(atom_type_matrix, bond_type_tensor, molecule_mask)

    def config_training(
        self,
        gen_optimizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
        lambda_rl: float = 0.0,
        reward_optimizer: Optional[torch.optim.Optimizer] = None,
        gen_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        disc_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        reward_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """Configure optimizers and schedulers for training."""
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.reward_optimizer = reward_optimizer
        self.lambda_rl = lambda_rl

        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.reward_scheduler = reward_scheduler

    def training_step(self, batch, reward_fn, pretrain=False):
        node_features, adjacency_matrix = batch
        batch_size = node_features.size(0)
        adjacency_matrix = adjacency_matrix.permute(0, 3, 1, 2)
        mask = (node_features.sum(-1) != 0).float()

        z = torch.randn(batch_size, self.gen_config.z_dim, device=node_features.device)
        fake_atom_logits, fake_bond_logits = self.gen(z)
        fake_atom = torch.softmax(fake_atom_logits, -1)
        fake_bond = torch.softmax(fake_bond_logits, 1)
        fake_mask = (fake_atom.argmax(-1) != 0).float()

        # === Discriminator update ===
        self.disc_optimizer.zero_grad()
        real_scores = self.disc(node_features, adjacency_matrix, mask)
        fake_scores = self.disc(fake_atom, fake_bond, fake_mask)
        wgan_loss = -(real_scores.mean() - fake_scores.mean())
        wgan_loss.backward()
        self.disc_optimizer.step()
        if self.disc_scheduler: self.disc_scheduler.step()

        # === Reward net update ===
        if self.reward_optimizer is not None:
            self.reward_optimizer.zero_grad()
            reward_targets = reward_fn(node_features, adjacency_matrix, mask)
            pred_rewards = self.reward_net(node_features, adjacency_matrix, mask)
            r_loss = torch.nn.functional.mse_loss(pred_rewards, reward_targets)
            r_loss.backward()
            self.reward_optimizer.step()
            if self.reward_scheduler: self.reward_scheduler.step()
        else:
            r_loss = torch.tensor(0.0, device=node_features.device)

        # === Generator update ===
        self.gen_optimizer.zero_grad()
        fake_atom_logits, fake_bond_logits = self.gen(z)
        fake_atom = torch.softmax(fake_atom_logits, -1)
        fake_bond = torch.softmax(fake_bond_logits, 1)
        fake_mask = (fake_atom.argmax(-1) != 0).float()
        fake_scores = self.disc(fake_atom, fake_bond, fake_mask)
        g_wgan_loss = -fake_scores.mean()
        if not pretrain and hasattr(self, 'lambda_rl') and self.lambda_rl > 0:
            with torch.no_grad():
                rewards = self.reward_net(fake_atom, fake_bond, fake_mask)
            rl_loss = -rewards.mean()
        else:
            rl_loss = torch.tensor(0.0, device=node_features.device)
        total_loss = g_wgan_loss + getattr(self, 'lambda_rl', 0.0) * rl_loss
        total_loss.backward()
        self.gen_optimizer.step()
        if self.gen_scheduler: self.gen_scheduler.step()

        return {
            'd_loss': wgan_loss.item(),
            'g_loss': g_wgan_loss.item(),
            'rl_loss': rl_loss.item(),
            'r_loss': r_loss.item() if self.reward_optimizer is not None else None
        }


    def train_epoch(self, dataloader, reward_fn, pretrain=False, log_interval=100):
        self.gen.train()
        self.disc.train()
        if self.reward_net: self.reward_net.train()
        for i, batch in enumerate(dataloader):
            result = self.training_step(batch, reward_fn, pretrain)
            if i % log_interval == 0:
                print({k: round(v, 5) for k, v in result.items()})





