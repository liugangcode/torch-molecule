import re
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, List, Type
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from .gpt import GPT
from .dataset import SmilesDataset
from ...base import BaseMolecularGenerator

@dataclass
class MolGPTMolecularGenerator(BaseMolecularGenerator):
    """
    This generator implements the molecular GPT model for generating molecules.
    
    The model uses a GPT-like architecture to learn the distribution of SMILES strings
    and generate new molecules. It supports conditional generation based on properties
    and/or molecular scaffolds.
    
    References
    ----------
    - MolGPT: Molecular Generation Using a Transformer-Decoder Model. Journal of Chemical 
      Information and Modeling. https://pubs.acs.org/doi/10.1021/acs.jcim.1c00600
    - Code: https://github.com/devalab/molgpt

    Parameters
    ----------
    num_layer : int, default=8
        Number of transformer layers in the model.
    num_head : int, default=8
        Number of attention heads in each transformer layer.
    hidden_size : int, default=256
        Dimension of the hidden representations.
    max_len : int, default=128
        Maximum length of SMILES strings.
    num_task : int, default=0
        Number of property prediction tasks for conditional generation. O for unconditional generation.
    use_scaffold : bool, default=False
        Whether to use scaffold conditioning.
    use_lstm : bool, default=False
        Whether to use LSTM for encoding scaffold.
    lstm_layers : int, default=0
        Number of LSTM layers if use_lstm is True.
    batch_size : int, default=64
        Batch size for training.
    epochs : int, default=1000
        Number of training epochs.
    learning_rate : float, default=3e-4
        Learning rate for optimizer.
    adamw_betas : Tuple[float, float], default=(0.9, 0.95)
        Beta parameters for AdamW optimizer.
    weight_decay : float, default=0.1
        Weight decay for optimizer.
    grad_norm_clip : float, default=1.0
        Gradient norm clipping value.
    verbose : bool, default=False
        Whether to display progress bars during training.
    """

    # Model parameters
    num_layer: int = 8
    num_head: int = 8
    hidden_size: int = 256
    max_len: int = 128
    
    # Conditioning parameters
    num_task: int = 0
    use_scaffold: bool = False
    use_lstm: bool = False
    lstm_layers: int = 0
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 1000
    learning_rate: float = 3e-4
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0

    verbose: bool = False
    # Attributes
    model_name: str = "MolGPTMolecularGenerator"
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    model_class: Type[GPT] = field(default=GPT, init=False)

    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__post_init__()
        self.vocab_size = None
        self.token_to_id = None
        self.id_to_token = None
        self.scaffold_maxlen = None
        self.pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        List[str]
            List of parameter names that can be used for model configuration.
        """
        return [
            # Model Hyperparameters
            "num_layer", "num_head", "hidden_size",
            # Conditioning parameters
            "num_task", "use_scaffold", "use_lstm", "lstm_layers",
            # post-initialization parameters
            "vocab_size", "token_to_id", "id_to_token", "scaffold_maxlen", "max_len",
            # Training Parameters
            "batch_size", "epochs", "learning_rate", "weight_decay", "grad_norm_clip",
            # Other Parameters
            "verbose", "model_name", "fitting_epoch", "fitting_loss", "device"
        ]
    
    def _initialize_model(
        self,
        model_class: Type[torch.nn.Module],
        checkpoint: Optional[Dict] = None
    ) -> torch.nn.Module:
        """Initialize the model with parameters or a checkpoint.
        
        Parameters
        ----------
        model_class : Type[torch.nn.Module]
            PyTorch module class to instantiate
        checkpoint : Optional[Dict], default=None
            Optional dictionary containing model checkpoint data
            
        Returns
        -------
        torch.nn.Module
            Initialized PyTorch model
        """
        model_params = self._get_model_params(checkpoint)
        self.model = model_class(**model_params)
        self.model = self.model.to(self.device)
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # get other params in checkpoint["hyperparameters"] but NOT in model_params
            other_params = {k: checkpoint["hyperparameters"][k] for k in checkpoint["hyperparameters"] if k not in model_params}
            # set other params in self
            for k, v in other_params.items():
                setattr(self, k, v)
        return self.model

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = [
            "vocab_size", "max_len", "num_task", "num_layer", "num_head", "hidden_size", "use_scaffold", "scaffold_maxlen", "use_lstm", "lstm_layers",
        ]        
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            return {k: checkpoint["hyperparameters"][k] for k in params}
    
        return {k: getattr(self, k) for k in params}

    def _setup_optimizers(self):
        train_config = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": self.adamw_betas
        }
        return self.model.configure_optimizers(train_config)

    def fit(self, X_train, y_train=None, X_scaffold=None):
        """
        Train the MolGPT model on SMILES strings.
        
        Parameters
        ----------
        X_train : List[str]
            List of SMILES strings for training
        y_train : Optional[List[float]]
            Optional list of property values for conditional generation
        X_scaffold : Optional[List[str]]
            Optional list of scaffold SMILES strings for conditional generation
            
        Returns
        -------
        self : MolGPTGenerator
            The fitted model
        """
        X_train, y_train = self._validate_inputs(X_train, y_train, num_task=self.num_task, return_rdkit_mol=False)

        # Calculate max length for padding
        lens = [len(self.regex.findall(i.strip())) for i in X_train]
        max_len = max(lens)

        if X_scaffold is not None:
            assert len(X_scaffold) == len(X_train), "X_scaffold and X_train must have the same length"
            assert self.use_scaffold, "use_scaffold must be True"
            X_scaffold, _ = self._validate_inputs(X_scaffold, num_task=self.num_task, return_rdkit_mol=False)
            scaffold_maxlen = max([len(self.regex.findall(i.strip())) for i in X_scaffold])
        else:
            scaffold_maxlen = 0
        
        self.scaffold_maxlen = scaffold_maxlen
        self.max_len = max_len

        # Create dataset
        train_dataset = SmilesDataset(
            X_train, 
            self.regex,
            max_len, 
            properties=y_train,
            scaffolds=X_scaffold,
            scaffold_maxlen=scaffold_maxlen,
        )
        
        # Save vocabulary
        self.vocab_size = train_dataset.vocab_size
        self.token_to_id = train_dataset.stoi
        self.id_to_token = train_dataset.itos
        
        # Initialize model
        self._initialize_model(self.model_class)
        self.model.initialize_parameters()
        optimizer = self._setup_optimizers()
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Training loop
        self.fitting_loss = []
        self.fitting_epoch = 0
        
        scaler = GradScaler()
        for epoch in range(self.epochs):
            train_losses = self._train_epoch(train_loader, optimizer, epoch, scaler)
            self.fitting_loss.append(np.mean(train_losses))

        self.fitting_epoch = epoch
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, epoch, scaler):
        self.model.train()
        losses = []
        
        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )
                
        for step, (x, y, prop, scaffold) in enumerate(iterator):
            x = x.to(self.device)
            y = y.to(self.device)
            prop = prop.to(self.device) if prop.numel() > 0 else None
            scaffold = scaffold.to(self.device) if scaffold.numel() > 0 else None
            
            optimizer.zero_grad()
            loss = self.model.compute_loss(x, targets=y, prop=prop, scaffold=scaffold)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Loss": f"{loss.item():.4f}"})
        
        return losses
        
    @torch.no_grad()
    def sample(self, x, steps, temperature=1.0, top_k=None, prop=None, scaffold=None):
        """
        Sample from the model given a context.
        
        Parameters
        ----------
        x : torch.Tensor
            Context tensor of shape (batch_size, seq_len)
        steps : int
            Number of steps to sample
        temperature : float
            Sampling temperature
        top_k : int
            Top-k sampling parameter
        prop : torch.Tensor
            Property conditioning tensor
        scaffold : torch.Tensor
            Scaffold conditioning tensor
            
        Returns
        -------
        torch.Tensor
            Generated sequences
        """
        model = self.model
        model.eval()
        
        for k in range(steps):
            # Get block size from model
            max_len = model.get_max_len()
            
            # Crop context if needed
            x_cond = x if x.size(1) <= max_len else x[:, -max_len:]
            
            # Forward pass
            logits, _ = model(x_cond, prop=prop, scaffold=scaffold)
            
            # Get logits for the next token and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling if specified
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            x = torch.cat((x, next_token), dim=1)
        
        return x
    
    def generate(self, n_samples=10, properties=None, scaffolds=None, max_len=None, temperature=1.0, top_k=10, starting_token='C'):
        """
        Generate molecules using the trained model.
        
        Parameters
        ----------
        n_samples : int, default=10
            Number of molecules to generate
        properties : Optional[List[List[float]]]
            Property values for conditional generation
        scaffolds : Optional[List[str]]
            Scaffold SMILES for conditional generation
        max_len : Optional[int]
            Maximum length of generated SMILES
        temperature : float, default=1.0
            Sampling temperature
        top_k : int, default=10
            Top-k sampling parameter
        starting_token : Optional[str]
            Starting token for generation (default is 'C')
            
        Returns
        -------
        List[str]
            List of generated SMILES strings
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before generating molecules")
        
        if max_len is None:
            max_len = self.max_len
        
        # Prepare property conditioning if provided
        if properties is not None:
            if len(properties) != n_samples:
                raise ValueError(f"Number of property values ({len(properties)}) must match n_samples ({n_samples})")
            prop_tensor = torch.tensor(properties, dtype=torch.float).to(self.device)
        else:
            prop_tensor = None
        
        # Prepare scaffold conditioning if provided
        if scaffolds is not None:
            if len(scaffolds) != n_samples:
                raise ValueError(f"Number of scaffolds ({len(scaffolds)}) must match n_samples ({n_samples})")
            
            # Tokenize scaffolds
            regex = re.compile(self.pattern)
            
            scaffold_tokens = []
            for scaffold in scaffolds:
                tokens = regex.findall(scaffold.strip())
                # Pad with '<' if needed
                tokens += ['<'] * (self.scaffold_maxlen - len(tokens))
                # Convert to indices
                scaffold_tokens.append([self.token_to_id.get(t, 0) for t in tokens])
            
            scaffold_tensor = torch.tensor(scaffold_tokens, dtype=torch.long).to(self.device)
        else:
            scaffold_tensor = None
        
        if starting_token in self.token_to_id:
            start_token = self.token_to_id[starting_token]
        else:
            warnings.warn(f"Starting token {starting_token} not found in vocabulary, using first token instead")
            start_token = 0
        
        context = torch.tensor([[start_token]] * n_samples, dtype=torch.long).to(self.device)
        
        # Sample from the model
        generated = self.sample(
            context, 
            steps=max_len-1,  # -1 because we already have one token
            temperature=temperature,
            top_k=top_k,
            prop=prop_tensor,
            scaffold=scaffold_tensor
        )
        
        # Convert to SMILES strings
        smiles_list = []
        for i in range(n_samples):
            # Convert indices to tokens
            tokens = [self.id_to_token[idx.item()] for idx in generated[i]]
            # Join tokens and remove padding
            smiles = ''.join(tokens).replace('<', '')
            smiles_list.append(smiles)
        
        return smiles_list
    