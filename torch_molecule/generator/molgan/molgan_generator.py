import torch
from typing import Optional, Union, List, Callable
from .molgan_model import MolGANModel
from .molgan_gen_disc import MolGANGeneratorConfig, MolGANDiscriminatorConfig
from torch_molecule.base.generator import BaseMolecularGenerator
from .molgan_dataset import MolGANDataset
from torch_molecule.utils import graph_to_smiles, graph_from_smiles




class MolGANGenerativeModel(BaseMolecularGenerator):

    """
    This generator implements the MolGAN model for molecular graph generation.

    The model uses a GAN like architecture with a generator and discriminator,
    combined with a reward network to optimize for desired molecular properties.
    The generator produces molecular graphs represented as adjacency matrices, with the discriminator 
    and reward network evaluating their validity and quality. The reward network can be trained to optimize 
    for specific chemical properties, such as drug-likeness or synthetic accessibility.


    References:
    ----------
        - De Cao, N., & Kipf, T. (2018). MolGAN: An implicit generative model for small molecular graphs.
        arXiv preprint arXiv:1805.11973. Link: https://arxiv.org/pdf/1805.11973

    Parameters:
    ----------
        MolGANGeneratorConfig : MolGANGeneratorConfig, optional
            Configuration for the generator network. If None, default values are used.

        MolGANDiscriminatorConfig : MolGANDiscriminatorConfig, Optional
            Configuration for the discriminator and reward network. If None, default values are used.

        Lambda_rl : float, Optional
            Weight for the reinforcement learning reward in the generator loss. Default is 0.25.

        device : Optional[Union[torch.device, str]], optional
            Device to run the model on. If None, defaults to CPU or GPU if available.

        model_name : str, Optional
            Name of the model. Default is "MolGANGenerativeModel".

    """

    def __init__(
        self,
        generator_config: Optional[MolGANGeneratorConfig] = None,
        discriminator_config: Optional[MolGANDiscriminatorConfig] = None,
        lambda_rl: float = 0.25,
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "MolGANGenerativeModel",
    ):
        super().__init__(device=device, model_name=model_name)

        # Initialize MolGAN model
        self.model = MolGANModel(
            generator_config=generator_config,
            discriminator_config=discriminator_config,
            reward_network_config=discriminator_config,
        ).to(self.device)

        self.lambda_rl = lambda_rl
        self.gen_optimizer = None
        self.disc_optimizer = None
        self.reward_optimizer = None
        self.gen_scheduler = None
        self.disc_scheduler = None
        self.reward_scheduler = None
        self.use_reward = False

        self.epoch = 0
        self.step = 0

    def training_config(
        self,
        lambda_rl: float = 0.25,
        reward_function: Optional[Callable] = None,
        gen_optimizer: Optional[torch.optim.Optimizer] = None,
        disc_optimizer: Optional[torch.optim.Optimizer] = None,
        reward_optimizer: Optional[torch.optim.Optimizer] = None,
        gen_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        disc_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        reward_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_reward: bool = True,
        epochs: int = 300,
        batch_size: int = 32,
        atom_types: List[str] = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H'],
        bond_types: List[str] = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
        max_num_atoms: int = 50,
    ):
        """
        Configure training parameters for MolGAN.

        Parameters:
        ----------
            lambda_rl : float, Optional
                Weight for the reinforcement learning reward in the generator loss. Default is 0.25.

            reward_function : Optional[Callable], optional

            gen_optimizer : torch.optim.Optimizer
                Optimizer for the generator network.

            disc_optimizer : torch.optim.Optimizer
                Optimizer for the discriminator network.

            reward_optimizer : Optional[torch.optim.Optimizer], optional
                Optimizer for the reward network. If None, the discriminator optimizer is used.

            gen_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
                Learning rate scheduler for the generator optimizer.

            disc_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
                Learning rate scheduler for the discriminator optimizer.

            reward_scheduler : Optional[torch.optim.lr_scheduler._LRScheduler], optional
                Learning rate scheduler for the reward optimizer.

            use_reward : bool, optional
                Whether to use the reward network during training. Default is True.

            epochs : int
                Number of training epochs. Default is 300.

            atom_types : List[str]
                List of atom types to consider in the molecular graphs. Default includes common organic atoms.

            bond_types : List[str]
                List of bond types to consider in the molecular graphs. Default includes common bond types.

            max_num_atoms : int
                Maximum number of atoms in the generated molecular graphs. Default is 50.
        """

        if gen_optimizer is None: gen_optimizer = torch.optim.Adam(self.model.gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
        if disc_optimizer is None: disc_optimizer = torch.optim.Adam(self.model.disc.parameters(), lr=0.0001, betas=(0.5, 0.999))
        if reward_optimizer is None: reward_optimizer = disc_optimizer

        self.model.config_training(
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            lambda_rl=lambda_rl,
            reward_optimizer=reward_optimizer,
            gen_scheduler=gen_scheduler,
            disc_scheduler=disc_scheduler,
            reward_scheduler=reward_scheduler,
        )
        self.lambda_rl = lambda_rl
        self.reward_function = reward_function
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.reward_optimizer = (
            reward_optimizer if reward_optimizer is not None else disc_optimizer
        )
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.reward_scheduler = reward_scheduler
        self.use_reward = use_reward
        self.epochs = epochs
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.max_num_atoms = max_num_atoms
        self.batch_size = batch_size


    def fit( self, X:List[str], y=None ) -> "BaseMolecularGenerator":
        """
        Fit the MolGAN model to the training data.

        Parameters:
        ----------
            X : List[str]
                List of SMILES strings representing the training molecules.

            y : Optional[np.ndarray], optional
                Optional array of target values for supervised training. Default is None. (Not used in MolGAN)
        """

        if self.gen_optimizer is None or self.disc_optimizer is None:
            # raise ValueError("Please configure training optimizers using `training_config()` before fitting.")
            # Set default optimizers if not configured
            self.training_config(
                gen_optimizer=torch.optim.Adam(self.model.gen.parameters(), lr=0.0001, betas=(0.5, 0.999)),
                disc_optimizer=torch.optim.Adam(self.model.disc.parameters(), lr=0.0001, betas=(0.5, 0.999)),
                lambda_rl=0.25,
                use_reward=True,
            )

        # Create a dataloader from the SMILES strings
        dataset = MolGANDataset(data=X, atom_types=self.atom_types, bond_types=self.bond_types, max_num_atoms=self.max_num_atoms, return_mol=False, device=self.device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            self.model.train_epoch(
                dataloader,
                reward_fn= None if not self.use_reward else self.reward_function
            )

        return self



