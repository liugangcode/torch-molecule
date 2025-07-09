import torch
import os, json
import numpy as np
import warnings
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List
from dataclasses import field

from torch_molecule.base.generator import BaseMolecularGenerator

# If for future compatibility, do ensure Configs are imported
from .generator import MolGANGenerator
from .discriminator import MolGANDiscriminator
from .rewards_molgan import RewardOracle
from .gan_utils import decode_smiles_from_graph
from .dataset import MolGraphDataset, molgan_collate_fn


# The actual MolGAN implementation
@dataclass
class MolGAN(BaseMolecularGenerator):
    """MolGAN implementation compatible with BaseMolecularGenerator interface."""

    model_name: str = field(default="MolGAN")

    def __init__(
        self,
        latent_dim: int = 56,
        hidden_dims_gen: List[int] = [128,128],
        hidden_dims_disc: List[int] = [128, 128],
        num_nodes: int = 9,
        tau: float = 1.0,
        num_atom_types: int = 5,
        num_bond_types: int = 4,
        use_reward: bool = False,
        device: Optional[str] = None
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dims_gen = hidden_dims_gen
        self.hidden_dims_disc = hidden_dims_disc
        self.num_nodes = num_nodes
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.use_reward = use_reward
        self.tau = tau

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.generator = MolGANGenerator(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims_gen,
            num_nodes=num_nodes,
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            tau=tau
        ).to(self.device)

        self.discriminator = MolGANDiscriminator(
            hidden_dims=hidden_dims_disc,
            num_nodes=num_nodes,
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types
        ).to(self.device)

        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

    def fit(
        self,
        X: List[str],
        y: Optional[np.ndarray] = None,
        reward: Optional[RewardOracle] = None,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> "MolGAN":
        """
        Fit the MolGAN model to a list of SMILES strings.

        Parameters
        ----------
        X : List[str]
            List of training SMILES strings.

        y : Optional[np.ndarray]
            Optional reward targets. (Unused if using oracle or no reward)

        epochs : int
            Number of training epochs.

        batch_size : int
            Batch size for training.

        Returns
        -------
        self : MolGAN
            The trained model.
        """

        if y is not None:
            warnings.warn("y is not used in MolGAN training. Use reward function instead.")

        from torch.utils.data import DataLoader

        dataset = MolGraphDataset(
            smiles_list=X,
            reward_function = reward if self.use_reward else None,
            max_nodes=self.num_nodes
        )

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=molgan_collate_fn,
            drop_last=True
        )

        self.generator.train()
        self.discriminator.train()

        for epoch in range(1, epochs + 1):
            epoch_d_loss = []
            epoch_g_loss = []

            for batch in train_loader:
                real_adj = batch["adj"].to(self.device)
                real_node = batch["node"].to(self.device)
                real_reward = batch["reward"].to(self.device)

                batch_size_actual = real_adj.size(0)
                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)

                # === Train Discriminator ===
                self.dis_opt.zero_grad()

                # Real loss
                d_real = self.discriminator(real_adj, real_node)
                d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))

                # Fake loss
                with torch.no_grad():
                    fake_adj, fake_node = self.generator(z)
                d_fake = self.discriminator(fake_adj, fake_node)
                d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.dis_opt.step()

                # === Train Generator ===
                self.gen_opt.zero_grad()
                fake_adj, fake_node = self.generator(z)
                d_fake = self.discriminator(fake_adj, fake_node)

                g_adv_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))

                # Reward-guided loss (optional)
                if self.use_reward and reward is not None:
                    with torch.no_grad():
                        rwd = reward(fake_adj, fake_node)  # [B]
                    g_rwd_loss = -rwd.mean()
                else:
                    g_rwd_loss = 0.0

                g_loss = g_adv_loss + g_rwd_loss
                g_loss.backward()
                self.gen_opt.step()

                epoch_d_loss.append(d_loss.item())
                epoch_g_loss.append(g_loss.item())

            print(f"[Epoch {epoch}/{epochs}] D_loss: {np.mean(epoch_d_loss):.4f} | G_loss: {np.mean(epoch_g_loss):.4f}")

        return self


    def generate(self, n_samples: int, **kwargs) -> List[str]:
        """
        Generate molecules from random latent vectors.

        Returns
        -------
        List[str] : Valid SMILES strings
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            adj, node = self.generator(z)
            smiles = decode_smiles_from_graph(adj, node)
            if smiles is None:
                return []
            return [s for s in smiles if s is not None]


    #  Initial implementation for saving and loading the model
    def save_pretrained(self, save_directory: str, configfile: Optional[str] = None):
        os.makedirs(save_directory, exist_ok=True)
        if configfile is None:
            configfile = "config.json"

        # Save model weights
        torch.save(self.generator.state_dict(), os.path.join(save_directory, "generator.pt"))
        torch.save(self.discriminator.state_dict(), os.path.join(save_directory, "discriminator.pt"))

        # Save config
        config = {
            "latent_dim": self.latent_dim,
            "hidden_dims_gen": self.hidden_dims_gen,
            "hidden_dims_disc": self.hidden_dims_disc,
            "num_nodes": self.num_nodes,
            "num_atom_types": self.num_atom_types,
            "num_bond_types": self.num_bond_types,
            "tau": self.tau,
            "use_reward": self.use_reward
        }
        with open(os.path.join(save_directory, configfile), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, load_directory: str, device: Optional[str] = None, configfile: str = "config.json") -> "MolGAN":
        with open(os.path.join(load_directory, configfile)) as f:
            config = json.load(f)

        model = cls(**config, device=device)
        model.generator.load_state_dict(torch.load(os.path.join(load_directory, "generator.pt"), map_location=device))
        model.discriminator.load_state_dict(torch.load(os.path.join(load_directory, "discriminator.pt"), map_location=device))
        return model

