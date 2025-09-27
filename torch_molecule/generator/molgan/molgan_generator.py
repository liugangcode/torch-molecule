import torch
import numpy as np
from typing import Optional, List, Union
from .molgan_model import MolGANModel
from .molgan_dataset import MolGANDataset
from torch_molecule.base.generator import BaseMolecularGenerator

class MolGANMolecularGenerator(BaseMolecularGenerator):
    """
    MolGAN model wrapper for standardized molecular generation API.
    Inherits fit/generate signature from BaseMolecularGenerator.
    """
    def __init__(
        self,
        generator, discriminator, reward_net=None,
        gen_config=None, disc_config=None,
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "MolGANMolecularGenerator"
    ):
        super().__init__(device=device, model_name=model_name)
        self.generator = generator
        self.discriminator = discriminator
        self.reward_net = reward_net
        self.gen_config = gen_config
        self.disc_config = disc_config

        self.device = device if device is not None else torch.device("cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        if self.reward_net: self.reward_net.to(self.device)
        self.is_fitted = False

    def fit(self, X: List[str], y: Optional[np.ndarray] = None, epochs=100, batch_size=32, **kwargs):
        """
        Train MolGAN on molecules (SMILES).
        """
        # 1. Prepare dataset (requires atom_types and bond_types in kwargs)
        atom_types = kwargs.get("atom_types")
        bond_types = kwargs.get("bond_types")
        max_num_atoms = kwargs.get("max_num_atoms", 50)
        dataset = MolGANDataset(X, atom_types, bond_types, max_num_atoms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        # 2. Set up optimizers/schedulers etc (credits: your config_training & usual setup)
        self.config_training(
            gen_optimizer=kwargs["gen_optimizer"],
            disc_optimizer=kwargs["disc_optimizer"],
            reward_optimizer=kwargs.get("reward_optimizer", None),
            gen_scheduler=kwargs.get("gen_scheduler", None),
            disc_scheduler=kwargs.get("disc_scheduler", None),
            reward_scheduler=kwargs.get("reward_scheduler", None),
        )
        # 3. Training loop (calls training_step as above)
        for epoch in range(epochs):
            for batch in dataloader:
                # Keep reward_fn optional for RL phase
                reward_fn = kwargs.get("reward_fn", None)
                pretrain = kwargs.get("pretrain", False)
                self.training_step(batch, reward_fn, pretrain)
        self.is_fitted = True
        return self

    def generate(self, n_samples: int, **kwargs) -> List[str]:
        """
        Generate n_samples molecules as SMILES.
        """
        self.generator.eval()
        zs = torch.randn(n_samples, self.generator.z_dim, device=self.device)
        with torch.no_grad():
            atom_logits, bond_logits = self.generator(zs)
            # [n_samples, max_num_atoms, num_atom_types], [n_samples, num_bond_types, max_num_atoms, max_num_atoms]
            atom_types = atom_logits.argmax(dim=-1).cpu().numpy()  # [n_samples, max_num_atoms]
            bond_types = bond_logits.argmax(dim=1).cpu().numpy()   # [n_samples, max_num_atoms, max_num_atoms]
        # Convert to SMILES (implement graph2smiles utility based on your atom_types/bond_types)
        smiles_strings = [
            graph2smiles(atom_types[i], bond_types[i], kwargs.get("atom_types"), kwargs.get("bond_types"))
            for i in range(n_samples)
        ]
        return smiles_strings
