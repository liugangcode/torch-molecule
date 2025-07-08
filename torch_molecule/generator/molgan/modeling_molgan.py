import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import MolGANGenerator
from .discriminator import MolGANDiscriminator
from .rewards import RewardNetwork


class MolGAN(nn.Module):
    """
    Full MolGAN model integrating:
    - Generator
    - Discriminator
    - Reward Network (oracle or neural)
    """

    def __init__(
        self,
        generator_config,
        discriminator_config,
        reward_config,
        use_reward=True,
        reward_lambda=1.0,
        device="cpu"
    ):
        super().__init__()
        self.device = device
        self.use_reward = use_reward
        self.reward_lambda = reward_lambda

        self.generator = MolGANGenerator(generator_config).to(device)
        self.discriminator = MolGANDiscriminator(discriminator_config).to(device)
        self.reward = RewardNetwork(**reward_config) if use_reward else None

        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=generator_config.get("lr", 1e-3))
        self.dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_config.get("lr", 1e-3))

    def generate(self, batch_size):
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        adj, node = self.generator(z)
        return adj, node

    def compute_rewards(self, smiles_list=None, adj=None, node=None):
        if self.reward is None:
            if adj is None or node is None:
                return torch.zeros(adj.size(0), device=self.device)
        return torch.tensor([
            self.reward(smiles=s) for s in smiles_list
        ], dtype=torch.float32, device=self.device) if smiles_list else self.reward(adj=adj, node=node)

    def decode_smiles(self, adj, node):
        """
        Convert batch of (adj, node) to SMILES strings
        This requires your graph_to_smiles function
        """
        from your_utils.graph_to_smiles import graph_to_smiles
        graphs = list(zip(node.cpu().numpy(), adj.cpu().numpy()))
        return graph_to_smiles(graphs, atom_decoder=["C", "N", "O", "F"])

    def fit(self, data_loader, epochs=10, log_every=1):
        for epoch in range(epochs):
            epoch_d_loss, epoch_g_loss, epoch_rewards = [], [], []
            for batch in data_loader:
                real_adj = batch["adj"].to(self.device)
                real_node = batch["node"].to(self.device)
                real_smiles = batch.get("smiles", None)

                # === Train Discriminator ===
                self.dis_opt.zero_grad()
                fake_adj, fake_node = self.generate(real_adj.size(0))

                real_logits = self.discriminator(real_adj, real_node)
                fake_logits = self.discriminator(fake_adj.detach(), fake_node.detach())
                d_loss = -torch.mean(real_logits) + torch.mean(fake_logits)
                d_loss.backward()
                self.dis_opt.step()

                # === Train Generator ===
                self.gen_opt.zero_grad()
                fake_logits = self.discriminator(fake_adj, fake_node)
                g_loss = -torch.mean(fake_logits)

                # Add reward loss if enabled
                if self.use_reward:
                    smiles_fake = self.decode_smiles(fake_adj, fake_node)
                    rewards = self.compute_rewards(smiles_list=smiles_fake)
                    reward_loss = -torch.mean(rewards)
                    g_loss = g_loss + self.reward_lambda * reward_loss
                else:
                    rewards = torch.zeros(real_adj.size(0))

                g_loss.backward()
                self.gen_opt.step()

                epoch_d_loss.append(d_loss.item())
                epoch_g_loss.append(g_loss.item())
                epoch_rewards.append(rewards.mean().item())

            if (epoch + 1) % log_every == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] D_loss: {sum(epoch_d_loss)/len(epoch_d_loss):.4f}, "
                      f"G_loss: {sum(epoch_g_loss)/len(epoch_g_loss):.4f}, "
                      f"Reward: {sum(epoch_rewards)/len(epoch_rewards):.4f}")
