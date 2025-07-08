from typing import Optional, List
import torch
import torch.nn as nn

from .generator import MolGANGenerator
from .discriminator import MolGANDiscriminator
from .rewards import RewardNetwork
from .gan_utils import decode_smiles, encode_smiles_to_graph
from ...utils.graph.graph_to_smiles import graph_to_smiles


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
            device="cpu"):
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

    def compute_rewards(
        self,
        smiles_list: Optional[List[str]] = None,
        adj: Optional[torch.Tensor] = None,
        node: Optional[torch.Tensor] = None,
    ):
        """
        Compute reward using the internal RewardNetwork, either from SMILES or from graph tensors.

        Parameters
        ----------
        smiles_list : List[str], optional
            List of SMILES strings to compute rewards for

        adj : Tensor [B, Y, N, N], optional
            Adjacency tensor

        node : Tensor [B, N, T], optional
            Node tensor

        Returns
        -------
        Tensor [B]
            Reward values
        """
        if self.reward is None:
            if adj is None or node is None:
                raise ValueError("Either smiles_list or (adj, node) must be provided for reward computation.")
            return torch.zeros(adj.size(0), device=self.device)

        if smiles_list is not None:
            adjs, nodes = [], []
            for smiles in smiles_list:
                try:
                    encoded_graph = encode_smiles_to_graph(
                        smiles,
                        atom_vocab=self.atom_decoder,
                        bond_types=self.bond_types,
                        max_nodes=self.max_nodes
                    )
                    if encoded_graph is None:
                        raise ValueError(f"Invalid SMILES: {smiles}")
                    a, n = encoded_graph
                    adjs.append(a)
                    nodes.append(n)
                except Exception:
                    # fallback to zeros if decoding fails
                    adjs.append(torch.zeros(len(self.bond_types), self.max_nodes, self.max_nodes))
                    nodes.append(torch.zeros(self.max_nodes, len(self.atom_decoder)))

            adj_batch = torch.stack(adjs).to(self.device)
            node_batch = torch.stack(nodes).to(self.device)
            return self.reward(adj_batch, node_batch)

        elif adj is not None and node is not None:
            return self.reward(adj.to(self.device), node.to(self.device))

        else:
            raise ValueError("Either smiles_list or (adj, node) must be provided for reward computation.")


    def fit(
        self,
        data_loader,
        epochs: int = 10,
        log_every: int = 1,
        reward_scale: float = 1.0
    ):
        """
        Train the MolGAN model using adversarial and (optional) reward-based learning.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader yielding batches of {"adj", "node", "smiles"} dictionaries.

        epochs : int
            Number of training epochs.

        log_every : int
            Frequency of logging losses.

        reward_scale : float
            Weight of reward loss in the generator's total loss.
        """
        self.generator.train()
        self.discriminator.train()
        if self.reward and hasattr(self.reward, 'neural') and self.reward.neural:
            self.reward.neural.eval()

        for epoch in range(epochs):
            d_losses, g_losses, reward_vals = [], [], []

            for batch in data_loader:
                real_adj = batch["adj"].to(self.device)      # [B, Y, N, N]
                real_node = batch["node"].to(self.device)    # [B, N, T]
                smiles = batch.get("smiles", None)

                batch_size = real_adj.size(0)

                # === Train Discriminator ===
                self.dis_opt.zero_grad()
                fake_adj, fake_node = self.generate(batch_size)

                real_logits = self.discriminator(real_adj, real_node)
                fake_logits = self.discriminator(fake_adj.detach(), fake_node.detach())

                d_loss = -torch.mean(real_logits) + torch.mean(fake_logits)
                d_loss.backward()
                self.dis_opt.step()

                # === Train Generator ===
                self.gen_opt.zero_grad()
                fake_logits = self.discriminator(fake_adj, fake_node)
                g_loss = -torch.mean(fake_logits)

                # === Add reward loss if applicable ===
                if self.use_reward:
                    rewards = self.compute_rewards(adj=fake_adj, node=fake_node)  # [B]
                    reward_loss = -rewards.mean()
                    g_loss += reward_scale * reward_loss
                    reward_vals.append(rewards.mean().item())
                else:
                    reward_vals.append(0.0)

                g_loss.backward()
                self.gen_opt.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            if (epoch + 1) % log_every == 0:
                print(f"[Epoch {epoch+1}/{epochs}] "
                      f"D_loss: {sum(d_losses)/len(d_losses):.4f} | "
                      f"G_loss: {sum(g_losses)/len(g_losses):.4f} | "
                      f"Reward: {sum(reward_vals)/len(reward_vals):.4f}")


    def fit(
        self,
        data_loader,
        epochs: int = 10,
        log_every: int = 1,
        reward_scale: float = 1.0
    ):
        """
        Train the MolGAN model using adversarial and (optional) reward-based learning.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader yielding batches of {"adj", "node", "smiles"} dictionaries.

        epochs : int
            Number of training epochs.

        log_every : int
            Frequency of logging losses.

        reward_scale : float
            Weight of reward loss in the generator's total loss.
        """
        self.generator.train()
        self.discriminator.train()
        if self.reward and hasattr(self.reward, 'neural') and self.reward.neural:
            self.reward.neural.eval()

        for epoch in range(epochs):
            d_losses, g_losses, reward_vals = [], [], []

            for batch in data_loader:
                real_adj = batch["adj"].to(self.device)
                real_node = batch["node"].to(self.device)
                smiles = batch.get("smiles", None)

                batch_size = real_adj.size(0)

                # === Train Discriminator ===
                self.dis_opt.zero_grad()
                fake_adj, fake_node = self.generate(batch_size)

                real_logits = self.discriminator(real_adj, real_node)
                fake_logits = self.discriminator(fake_adj.detach(), fake_node.detach())

                d_loss = -torch.mean(real_logits) + torch.mean(fake_logits)
                d_loss.backward()
                self.dis_opt.step()

                # === Train Generator ===
                self.gen_opt.zero_grad()
                fake_logits = self.discriminator(fake_adj, fake_node)
                g_loss = -torch.mean(fake_logits)

                # === Add reward loss if applicable ===
                if self.use_reward:
                    # Convert fake graphs to SMILES if reward expects SMILES
                    if self.reward.oracle is not None:
                        smiles_fake = self.decode_smiles(fake_adj, fake_node)
                        rewards = self.compute_rewards(smiles_list=smiles_fake)
                    else:
                        rewards = self.compute_rewards(adj=fake_adj, node=fake_node)

                    reward_loss = -rewards.mean()
                    g_loss += reward_scale * reward_loss
                    reward_vals.append(rewards.mean().item())
                else:
                    reward_vals.append(0.0)

                g_loss.backward()
                self.gen_opt.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            if (epoch + 1) % log_every == 0:
                print(f"[Epoch {epoch+1}/{epochs}] "
                      f"D_loss: {sum(d_losses)/len(d_losses):.4f} | "
                      f"G_loss: {sum(g_losses)/len(g_losses):.4f} | "
                      f"Reward: {sum(reward_vals)/len(reward_vals):.4f}")

