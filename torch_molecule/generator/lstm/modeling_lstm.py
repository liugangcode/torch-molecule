from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable, Type
from dataclasses import dataclass, field
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from .lstm import LSTM
from .utils import canonicalize
from .action_sampler import ActionSampler
from .smiles_char_dict import SmilesCharDictionary
from ...base import BaseMolecularGenerator

@dataclass
class LSTMMolecularGenerator(BaseMolecularGenerator):
    """LSTM-based molecular generator.
    
    This generator implements an LSTM architecture for molecular generation. 
    When conditions (y values) are provided during training, they are used to initialize the hidden 
    and cell states of the LSTM.
    
    Parameters
    ----------
    num_task : int, default=0
        Number of tasks for conditional generation (0 means unconditional generation).
    max_len : int, default=100
        Maximum length of the SMILES strings.
    num_layer : int, default=3
        Number of LSTM layers.
    hidden_size : int, default=512
        Dimension of hidden states in LSTM.
    dropout : float, default=0.2
        Dropout probability for regularization.
    batch_size : int, default=128
        Batch size for training.
    epochs : int, default=10000
        Maximum number of training epochs.
    learning_rate : float, default=0.0002
        Learning rate for optimizer.
    weight_decay : float, default=0.0
        L2 regularization factor.
    use_lr_scheduler : bool, default=False
        Whether to use learning rate scheduler.
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate when using scheduler (if True).
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced (if True).
    grad_norm_clip : Optional[float], default=None
        Maximum norm for gradient clipping. None means no clipping.
    verbose : bool, default=False
        Whether to print progress during training.
    """
    num_task: int = 0
    # LSTM parameters
    max_len: int = 100
    num_layer: int = 3
    hidden_size: int = 512
    dropout: float = 0.2

    # Training parameters
    batch_size: int = 128
    epochs: int = 10000
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    # Scheduler parameters
    use_lr_scheduler: bool = False
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    grad_norm_clip: Optional[float] = None
    # Other parameters
    verbose: bool = False

    # attributes
    model_name: str = "LSTMMolecularGenerator"
    fitting_loss: List[float] = field(default_factory=list, init=False)
    fitting_epoch: int = field(default=0, init=False)
    input_size: int = field(init=False, default=None)
    output_size: int = field(init=False, default=None)
    model_class: Type[LSTM] = field(default=LSTM, init=False)
    
    def __post_init__(self):
        super().__post_init__()
        self.tokenizer = SmilesCharDictionary()
        self.input_size = len(self.tokenizer.char_idx)
        self.output_size = len(self.tokenizer.char_idx)

    @staticmethod
    def _get_param_names() -> List[str]:
        return [
            "num_task", "max_len", "num_layer", "hidden_size", "dropout",
            "batch_size", "epochs", "learning_rate", "weight_decay",
            "use_lr_scheduler", "scheduler_factor", "scheduler_patience",
            "grad_norm_clip", "verbose", "input_size", "output_size", 'model_name'
        ]
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = [
            "num_task", "input_size", "hidden_size", "output_size", "num_layer", "dropout",
        ]
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            return {k: checkpoint["hyperparameters"][k] for k in params}
    
        return {k: getattr(self, k) for k in params}

    def _convert_to_pytorch_data(self, X, y=None):
        # filter valid smiles strings
        valid_smiles = []
        for s in X:
            s = s.strip()
            if self.tokenizer.allowed(s) and len(s) <= self.max_len:
                valid_smiles.append(s)
            else:
                valid_smiles.append('C')  # default placeholder
        
        # max len + two chars for start token 'Q' and stop token '\n'
        max_seq_len = self.max_len + 2
        # allocate the zero matrix to be filled
        seqs = np.zeros((len(valid_smiles), max_seq_len), dtype=np.int32)
        
        for i, mol in enumerate(valid_smiles):
            enc_smi = self.tokenizer.BEGIN + self.tokenizer.encode(mol) + self.tokenizer.END
            for c in range(len(enc_smi)):
                seqs[i, c] = self.tokenizer.char_idx[enc_smi[c]]
        
        seqs_tensor = torch.from_numpy(seqs).long()
        inp = seqs_tensor[:, :-1]
        target = seqs_tensor[:, 1:]
        bsz = inp.size(0)
        if y is not None:
            assert len(y) == bsz
            y = torch.tensor(y)
            return TensorDataset(inp, target, y)
        else:
            target_y = torch.zeros(bsz, 1)
            return TensorDataset(inp, target, target_y)

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Setup optimization components including optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = None
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=1e-6,
                cooldown=0,
                eps=1e-8,
            )

        return optimizer, scheduler
    
    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]] = None,
    ) -> "LSTMMolecularGenerator":
        X_train, y_train = self._validate_inputs(X_train, y_train, num_task=self.num_task, return_rdkit_mol=False)
        self._initialize_model(self.model_class)
        self.model.initialize_parameters()
        optimizer, scheduler = self._setup_optimizers()

        train_dataset = self._convert_to_pytorch_data(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.fitting_loss = []
        self.fitting_epoch = 0
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            train_losses = self._train_epoch(train_loader, optimizer, epoch, criterion)
            self.fitting_loss.append(np.mean(train_losses).item())
            if scheduler:
                scheduler.step(np.mean(train_losses).item())

        self.fitting_epoch = epoch
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, epoch, criterion):
        self.model.train()
        losses = []
        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )
        for step, batched_data in enumerate(iterator):
            for i in range(len(batched_data)):
                batched_data[i] = batched_data[i].to(self.device)
            optimizer.zero_grad()

            loss = self.model.compute_loss(batched_data, criterion)
            loss.backward()
            if self.grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            optimizer.step()
            losses.append(loss.item())
            
            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Loss": f"{loss.item():.4f}"})
            
        return losses

    def generate(
        self, 
        labels: Optional[Union[List[List], np.ndarray]] = None,
        batch_size: int = 32
    ) -> List[str]:
        """Generate molecules using LSTM.
        
        Parameters
        ----------
        labels : Optional[Union[List[List], np.ndarray]]
            Target property values for conditional generation.
        batch_size : int
            Number of molecules to generate.
            
        Returns
        -------
        List[str]
            Generated molecules as SMILES strings.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before generating molecules.")
        
        if labels is not None and batch_size != len(labels):
            print(f"batch_size is not equal to the length of labels, using the length of labels: {len(labels)}")
            batch_size = len(labels)

        # Convert properties to 2D tensor if needed
        if isinstance(labels, list):
            labels = torch.tensor(labels)
        elif isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        if labels is not None and labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        elif labels is None:
            labels = torch.zeros(batch_size, 1)
        labels = labels.to(self.device)
        
        all_generated_mols = []
        sampler = ActionSampler(max_batch_size=batch_size, max_seq_length=self.max_len, device=self.device)
        self.model.eval()
        with torch.no_grad():
            indices = sampler.sample(self.model, num_samples=batch_size, target=labels)
            samples = self.tokenizer.matrix_to_smiles(indices)
            canonicalized_smiles = [canonicalize(smiles, include_stereocenters=True) for smiles in samples]
            all_generated_mols.extend(canonicalized_smiles)

        return all_generated_mols
    
