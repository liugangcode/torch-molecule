from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Type
import numpy as np

import torch

from .jtnn_vae import JTNNVAE
from .jtnn.mol_tree import MolTree
from .jtnn.vocab import Vocab
from .jtnn.datautils import MolTreeFolder
from ...base import BaseMolecularGenerator

@dataclass
class JTVAEMolecularGenerator(BaseMolecularGenerator):
    """ 
    JT-VAE-based molecular generator. Implemented for unconditional moleculargeneration.     

    References
    ----------
    - Junction Tree Variational Autoencoder for Molecular Graph Generation. ICML 2018. https://arxiv.org/pdf/1802.04364
    - Code: https://github.com/kamikaze0923/jtvae

    Parameters
    ----------
    hidden_size : int, default=450
        Dimension of hidden layers in the model.
    latent_size : int, default=56
        Dimension of the latent space.
    depthT : int, default=20
        Depth of the tree encoder.
    depthG : int, default=3
        Depth of the graph decoder.
    batch_size : int, default=32
        Number of samples per batch during training.
    epochs : int, default=20
        Number of epochs to train the model.
    learning_rate : float, default=0.003
        Initial learning rate for the optimizer.
    weight_decay : float, default=0.0
        L2 regularization factor.
    grad_norm_clip : Optional[float], default=None
        Maximum norm for gradient clipping. None means no clipping.
    beta : float, default=0.0
        Initial KL divergence weight for VAE training.
    step_beta : float, default=0.002
        Step size for KL annealing.
    max_beta : float, default=1.0
        Maximum value for KL weight.
    warmup : int, default=40000
        Number of steps for KL annealing warmup.
    use_lr_scheduler : bool, default=True
        Whether to use learning rate scheduling.
    anneal_rate : float, default=0.9
        Learning rate annealing factor.
    anneal_iter : int, default=40000
        Number of iterations between learning rate updates.
    kl_anneal_iter : int, default=2000
        Number of iterations between KL weight updates.
    verbose : bool, default=False
        Whether to print detailed training information.
    """
    # Model parameters
    hidden_size: int = 450
    latent_size: int = 56
    depthT: int = 20
    depthG: int = 3

    # Training parameters
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    grad_norm_clip: Optional[float] = None
    
    # KL annealing parameters
    beta: float = 0.0
    step_beta: float = 0.002
    max_beta: float = 1.0
    warmup: int = 40000

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    anneal_rate: float = 0.9
    anneal_iter: int = 40000
    kl_anneal_iter: int = 2000

    # Other parameters
    verbose: bool = False

    # attributes
    model_name: str = "JTVAEMolecularGenerator"
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    model_class: Type[JTNNVAE] = field(default=JTNNVAE, init=False)
    
    def __post_init__(self):
        super().__post_init__()
        self.vocab = None

    @staticmethod
    def _get_param_names() -> List[str]:
        return [
            "hidden_size", "latent_size", "depthT", "depthG",
            "batch_size", "epochs", "learning_rate", "weight_decay",
            "beta", "step_beta", "max_beta", "warmup",
            "use_lr_scheduler", "anneal_rate", "anneal_iter", "kl_anneal_iter",
            "verbose", 'model_name', 'vocab'
        ]
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = [
            "vocab", "hidden_size", "latent_size", "depthT", "depthG"
        ]
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            return {k: checkpoint["hyperparameters"][k] for k in params}
    
        return {k: getattr(self, k) for k in params}

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimization components including optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.anneal_rate)

        return optimizer, scheduler

    def _extract_vocab(self, X):
        cset = set()
        for ii, smiles in enumerate(X):
            try:
                mol = MolTree(smiles)
                for c in mol.nodes:
                    cset.add(c.smiles)
            except Exception as e:
                print(f'Error {e} in extracting vocab for smiles: {smiles}')
                pass
        vocab = list(cset)
        return vocab

    def _convert_to_tensor(self, X):
        all_data = []
        for smiles in X:
            try:
                mol_tree = MolTree(smiles)
                mol_tree.recover()
                mol_tree.assemble()
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)

                del mol_tree.mol
                for node in mol_tree.nodes:
                    del node.mol
            except Exception as e:
                print(f'Error {e} in tensorizing smiles: {smiles}')
                mol_tree = None
            all_data.append(mol_tree)
        return all_data

    def fit(
        self,
        X_train: List[str],
    ) -> "JTVAEMolecularGenerator":

        X_train, _ = self._validate_inputs(X_train, None, num_task=0, return_rdkit_mol=False)
        vocab = self._extract_vocab(X_train)
        vocab = Vocab(vocab)
        self.vocab = vocab

        self._initialize_model(self.model_class)
        self.model.initialize_parameters()
        optimizer, scheduler = self._setup_optimizers()

        train_dataset = self._convert_to_tensor(X_train)
        train_dataset = list(filter(lambda x: x is not None, train_dataset))
        train_loader = MolTreeFolder(train_dataset, vocab, self.batch_size, num_workers=0)
        step_len = len(X_train) // self.batch_size
        self.fitting_loss = []
        self.fitting_epoch = 0
        total_step = 0
        for epoch in range(self.epochs):
            train_losses, total_step = self._train_epoch(train_loader, optimizer, scheduler, epoch, total_step, step_len)
            self.fitting_loss.append(np.mean(train_losses).item())

        self.fitting_epoch = epoch
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, scheduler, epoch, total_step, step_len):
        self.model.train()
        losses = []
        iterator = (
            tqdm(train_loader, desc="Training", leave=False, total=step_len)
            if self.verbose
            else train_loader
        )
        for step, batched_data in enumerate(iterator):
            total_step += 1
            optimizer.zero_grad()

            word_loss, topo_loss, assm_loss, kl_div = self.model.compute_loss(batched_data)
            total_loss = word_loss + topo_loss + assm_loss + kl_div * self.beta
            total_loss.backward()
            if self.grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            optimizer.step()
            losses.append(total_loss.item())

            if total_step % self.anneal_iter == 0:
                scheduler.step()
                
            if total_step % self.kl_anneal_iter == 0 and total_step >= self.warmup:
                self.beta = min(self.max_beta, self.beta + self.step_beta)

            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Loss": f"{total_loss.item():.4f}, word_loss: {word_loss.item():.4f}, topo_loss: {topo_loss.item():.4f}, assm_loss: {assm_loss.item():.4f}, kl_div: {kl_div.item():.4f}"})
            
        return losses, total_step

    def generate(
        self, 
        batch_size: int = 32
    ) -> List[str]:
        """Generate molecules using JT-VAE.
        
        Parameters
        ----------  
        batch_size : int
            Number of molecules to generate.
            
        Returns
        -------
        List[str]
            Generated molecules as SMILES strings.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before generating molecules.")
        
        return self.model.sample_prior(self.device)
    
