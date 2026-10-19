"""GenMol molecular generation with masked discrete diffusion."""

import math
import pickle
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...base import BaseMolecularGenerator
from .diffusion import MaskedDiffusion
from .model import GenMolModel
from .tokenizer import (
    GenMolTokenizer,
    decode_safe,
    encode_smiles,
    encode_scaffold,
    filter_by_substructure,
)


class GenMolMolecularGenerator(BaseMolecularGenerator):
    """Generate molecules using GenMol V1 masked diffusion over SAFE sequences.

    Parameters
    ----------
    num_layer : int, default=12
        Number of bidirectional transformer layers.
    num_head : int, default=12
        Number of attention heads; must divide hidden_size.
    hidden_size : int, default=768
        Transformer hidden dimension.
    intermediate_size : int, default=3072
        Feed-forward hidden dimension.
    max_len : int, default=256
        Maximum token length including beginning and end tokens.
    dropout, attention_dropout : float, default=0.1
        Hidden and attention dropout probabilities.
    hidden_act : str, default="gelu"
        BERT feed-forward activation: gelu, relu, silu, or gelu_new.
    layer_norm_eps : float, default=1e-12
        Layer normalization epsilon.
    initializer_range : float, default=0.02
        Standard deviation of the weight initializer.
    type_vocab_size : int, default=2
        BERT token-type vocabulary size.
    ema_decay : float, default=0.9999
        Exponential moving average decay. Zero disables EMA.
    antithetic_sampling : bool, default=True
        Stratify sampled masking times within each batch.
    sampling_eps : float, default=0.001
        Lower time bound for antithetic sampling.
    global_mean_loss : bool, default=True
        Average loss over tokens; False gives each molecule equal weight.
    batch_size : int, default=64
        Training and generation batch size.
    epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=0.0003
        AdamW learning rate after warmup.
    adamw_betas : tuple, default=(0.9, 0.999)
        AdamW moment decay rates.
    adamw_eps : float, default=1e-8
        AdamW denominator epsilon.
    weight_decay : float, default=0.0
        AdamW weight decay.
    warmup_steps : int, default=2500
        Linear learning-rate warmup steps; zero disables warmup.
    grad_clip_value : float or None, default=1.0
        Maximum gradient norm; None disables clipping.
    verbose : str, default="none"
        One of none, progress_bar, or print_statement.
    device : torch.device or str, optional
        CPU or CUDA device; defaults to CUDA when available.
    model_name : str, default="GenMolMolecularGenerator"
        Estimator name.

    References
    ----------
    Paper: https://proceedings.mlr.press/v267/lee25o.html
    Code: https://github.com/NVIDIA-BioNeMo/genmol
    """

    def __init__(
        self,
        num_layer: int = 12,
        num_head: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        max_len: int = 256,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        type_vocab_size: int = 2,
        ema_decay: float = 0.9999,
        antithetic_sampling: bool = True,
        sampling_eps: float = 1e-3,
        global_mean_loss: bool = True,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 3e-4,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 2500,
        grad_clip_value: Optional[float] = 1.0,
        verbose: str = "none",
        *,
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "GenMolMolecularGenerator",
    ):
        super().__init__(device=device, model_name=model_name, verbose=verbose)
        self.num_layer = num_layer
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_len = max_len
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.ema_decay = ema_decay
        self.antithetic_sampling = antithetic_sampling
        self.sampling_eps = sampling_eps
        self.global_mean_loss = global_mean_loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.adamw_betas = adamw_betas
        self.adamw_eps = adamw_eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.grad_clip_value = grad_clip_value
        self.fitting_loss = []
        self.fitting_epoch = 0
        self.sequence_lengths_ = []
        self.tokenizer_json_ = None
        self.tokenizer = None
        self.model_class = GenMolModel
        self._validate_hyperparameters()

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get the estimator parameters and fitted metadata."""
        return [
            "num_layer",
            "num_head",
            "hidden_size",
            "intermediate_size",
            "max_len",
            "dropout",
            "attention_dropout",
            "hidden_act",
            "layer_norm_eps",
            "initializer_range",
            "type_vocab_size",
            "ema_decay",
            "antithetic_sampling",
            "sampling_eps",
            "global_mean_loss",
            "batch_size",
            "epochs",
            "learning_rate",
            "adamw_betas",
            "adamw_eps",
            "weight_decay",
            "warmup_steps",
            "grad_clip_value",
            "sequence_lengths_",
            "tokenizer_json_",
            "fitting_loss",
            "fitting_epoch",
            "verbose",
            "device",
            "model_name",
        ]

    def _validate_hyperparameters(self):
        for name in [
            "num_layer",
            "num_head",
            "hidden_size",
            "intermediate_size",
            "max_len",
            "type_vocab_size",
            "batch_size",
            "epochs",
        ]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_len < 3 or self.hidden_size % self.num_head:
            raise ValueError("max_len must be at least 3 and hidden_size divisible by num_head.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative.")
        for name in ["dropout", "attention_dropout", "ema_decay", "sampling_eps"]:
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1).")

    def set_params(self, **params) -> "GenMolMolecularGenerator":
        previous = self.__dict__.copy()
        model_params = self._get_model_params()
        try:
            if "device" in params:
                params["device"] = torch.device(
                    params["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
                )
            super().set_params(**params)
            self._validate_hyperparameters()
            if any(key in model_params and value != previous[key] for key, value in params.items()):
                self.is_fitted_ = False
            if "device" in params and self.model is not None:
                self.model.to(self.device)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise
        return self

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = [
            "num_layer",
            "num_head",
            "hidden_size",
            "intermediate_size",
            "max_len",
            "dropout",
            "attention_dropout",
            "hidden_act",
            "layer_norm_eps",
            "initializer_range",
            "type_vocab_size",
            "ema_decay",
        ]
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            model_params = {name: checkpoint["hyperparameters"][name] for name in params}
        else:
            model_params = {name: getattr(self, name) for name in params}
        model_params.update(
            vocab_size=len(self.tokenizer) if self.tokenizer is not None else None,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )
        return model_params

    def _initialize_model(
        self,
        model_class: Type[torch.nn.Module],
        checkpoint: Optional[Dict] = None,
    ) -> torch.nn.Module:
        """Initialize the tokenizer and backbone from estimator or checkpoint parameters."""
        tokenizer_json = self.tokenizer_json_
        if checkpoint is not None:
            tokenizer_json = checkpoint["hyperparameters"]["tokenizer_json_"]
        self.tokenizer = GenMolTokenizer(tokenizer_json)
        self.tokenizer_json_ = self.tokenizer.to_json()
        return super()._initialize_model(model_class, checkpoint)

    def _convert_to_pytorch_data(self, X: List[str], y=None) -> List[List[int]]:
        """Convert SMILES to token sequences for the training data loader."""
        tokenizer = GenMolTokenizer()
        sequences = [tokenizer.encode(encode_smiles(s), max_len=self.max_len) for s in X]
        self.sequence_lengths_ = [len(sequence) for sequence in sequences]
        self.tokenizer = tokenizer
        self.tokenizer_json_ = tokenizer.to_json()
        return sequences

    def _diffusion(self):
        return MaskedDiffusion(
            self.tokenizer.mask_token_id,
            len(self.tokenizer),
            antithetic_sampling=self.antithetic_sampling,
            sampling_eps=self.sampling_eps,
        )

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Set up AdamW and the native linear warmup schedule."""
        optimizer = torch.optim.AdamW(
            self.model.backbone.parameters(),
            lr=self.learning_rate,
            betas=self.adamw_betas,
            eps=self.adamw_eps,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, step / max(1, self.warmup_steps)) if self.warmup_steps else 1.0,
        )
        return optimizer, scheduler

    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]] = None,
    ) -> "GenMolMolecularGenerator":
        """Train a fresh model on SMILES strings.

        Parameters
        ----------
        X_train : List[str]
            Training molecules as SMILES strings. Polymer attachment atoms (``*``)
            are not preserved in generated molecules.
        y_train : None
            Property labels are unsupported.

        Returns
        -------
        GenMolMolecularGenerator
            The fitted estimator.
        """
        if y_train is not None:
            raise ValueError("GenMol fit does not accept property labels (y_train).")
        X_train, y_train = self._validate_inputs(X_train, y_train, return_rdkit_mol=False)
        if any("*" in smiles for smiles in X_train):
            warnings.warn(
                "GenMol treats '*' atoms as attachment points; generated molecules "
                "do not preserve polymer endpoints.",
                UserWarning,
                stacklevel=2,
            )
        train_dataset = self._convert_to_pytorch_data(X_train, y_train)
        self.is_fitted_ = False
        self._initialize_model(self.model_class)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.tokenizer.pad,
        )
        optimizer, scheduler = self._setup_optimizers()
        self.fitting_loss = []
        self.fitting_epoch = 0
        global_pbar = None
        if self.verbose == "progress_bar":
            global_pbar = tqdm(total=self.epochs * len(train_loader), desc="Training Progress")
        try:
            for epoch in range(self.epochs):
                train_losses = self._train_epoch(
                    train_loader, optimizer, epoch, global_pbar, scheduler=scheduler
                )
                self.fitting_loss.append(np.mean(train_losses).item())
                self.fitting_epoch = epoch
        finally:
            if global_pbar is not None:
                global_pbar.close()
        self.model.eval()
        self.is_fitted_ = True
        return self

    def _train_epoch(
        self, train_loader, optimizer, epoch, global_pbar=None, scheduler=None
    ) -> List[float]:
        """Train one epoch and return one loss value per batch."""
        self.model.train()
        diffusion = self._diffusion()
        losses = []
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = self.model.compute_loss(batch, diffusion, global_mean=self.global_mean_loss)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()
            if self.model.ema is not None:
                self.model.ema.update(self.model.backbone.parameters())
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
            log_dict = {
                "Epoch": f"{epoch + 1}/{self.epochs}",
                "Step": f"{step + 1}/{len(train_loader)}",
                "Loss": f"{loss.item():.4f}",
            }
            if global_pbar is not None:
                global_pbar.set_postfix(log_dict)
                global_pbar.update(1)
            if self.verbose == "print_statement":
                print(log_dict)
        return losses

    def _check_fitted(self):
        if not self.is_fitted_ or self.model is None:
            raise ValueError("GenMol must be fitted or loaded before generation.")

    def _masked_inputs(self, prefix, count, min_new_tokens):
        if len(prefix) + min_new_tokens > self.max_len:
            raise ValueError("The scaffold and min_new_tokens exceed max_len.")
        sequences = []
        for _ in range(count):
            length = max(random.choice(self.sequence_lengths_) - len(prefix), min_new_tokens)
            length = min(length, self.max_len - len(prefix))
            sequences.append(prefix[:-1] + [self.tokenizer.mask_token_id] * length + prefix[-1:])
        return self.tokenizer.pad(sequences)["input_ids"].to(self.device)

    @torch.no_grad()
    def _sample_tokens(self, tokens, temperature, randomness, context_mask_ratio, guidance_scale):
        diffusion = self._diffusion()
        steps = max(diffusion.get_num_steps_confidence(tokens), 2)
        attention_mask = tokens != self.tokenizer.pad_token_id
        for step in range(steps):
            logits = self.model(tokens, attention_mask)
            if context_mask_ratio and guidance_scale:
                poor = tokens.clone()
                context = torch.ones_like(tokens[0], dtype=torch.bool)
                for token_id in [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.pad_token_id,
                ]:
                    context &= tokens[0] != token_id
                indices = context.nonzero(as_tuple=True)[0].tolist()
                selected = random.sample(indices, int(len(indices) * context_mask_ratio))
                poor[:, selected] = self.tokenizer.mask_token_id
                logits = guidance_scale * logits + (1 - guidance_scale) * self.model(
                    poor, attention_mask
                )
            tokens = diffusion.step_confidence(
                logits,
                tokens,
                step,
                steps,
                temperature=temperature,
                randomness=randomness,
            )
        return tokens

    def generate(
        self,
        n_samples: int = 10,
        *,
        scaffold: Optional[str] = None,
        temperature: float = 0.8,
        randomness: float = 0.5,
        min_new_tokens: int = 40,
        max_attempts: int = 10,
        guidance_scale: float = 2.0,
        context_mask_ratio: float = 0.0,
        fix: bool = True,
    ) -> List[str]:
        """Generate SMILES, optionally containing a supplied scaffold.

        ``n_samples`` is the requested number of valid outputs, which may include duplicates.
        ``min_new_tokens`` controls the minimum masked extension length. ``max_attempts``
        bounds rejection sampling per output batch; exhaustion warns and returns the
        available molecules, which may be an empty list.
        ``temperature`` and ``randomness`` control token and confidence sampling.
        ``context_mask_ratio`` enables molecular context guidance for scaffold generation.
        ``fix`` enables the original SAFE fragment repair during decoding.
        """
        self._check_fitted()
        if n_samples < 0 or min_new_tokens < 1 or max_attempts < 1:
            raise ValueError("n_samples must be >= 0; min_new_tokens and max_attempts >= 1.")
        if not 0 <= context_mask_ratio <= 1:
            raise ValueError("context_mask_ratio must be in [0, 1].")
        if n_samples == 0:
            return []
        if scaffold is None:
            prefix = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]
        else:
            molecule = Chem.MolFromSmiles(scaffold)
            if molecule is None:
                raise ValueError("Invalid scaffold SMILES.")
            if len(Chem.GetMolFrags(molecule)) != 1:
                raise ValueError("scaffold must be a single connected fragment.")
            if all(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
                raise ValueError("The scaffold must contain at least one non-dummy atom.")
            prefix = self.tokenizer.encode(encode_scaffold(scaffold), max_len=self.max_len)
        generated = []
        was_training = self.model.training
        self.model.eval()
        try:
            with self.model.averaged_parameters():
                for _ in range(max_attempts * math.ceil(n_samples / self.batch_size)):
                    count = min(n_samples - len(generated), self.batch_size)
                    inputs = self._masked_inputs(prefix, count, min_new_tokens)
                    tokens = self._sample_tokens(
                        inputs, temperature, randomness, context_mask_ratio, guidance_scale
                    )
                    for text in self.tokenizer.batch_decode(tokens.tolist()):
                        smiles = decode_safe(text, fix=fix)
                        if not smiles:
                            continue
                        smiles = sorted(smiles.split("."), key=len)[-1]
                        molecule = Chem.MolFromSmiles(smiles)
                        if (
                            molecule is not None
                            and molecule.GetNumAtoms()
                            and (scaffold is None or filter_by_substructure([smiles], scaffold))
                        ):
                            generated.append(Chem.MolToSmiles(molecule))
                    if len(generated) == n_samples:
                        return generated
        finally:
            self.model.train(was_training)
        warnings.warn(
            f"GenMol generated {len(generated)} of {n_samples} valid molecules within "
            f"max_attempts={max_attempts}. Returning the available molecules.",
            UserWarning,
            stacklevel=2,
        )
        return generated

    def load_from_local(self, path: str) -> None:
        """Restore a torch-molecule checkpoint on the selected device."""
        super().load_from_local(str(path))
        self.model.eval()

    def save_to_local(self, path: str) -> None:
        """Save weights, EMA, tokenizer, and fitted settings."""
        return super().save_to_local(str(path))

    def load_from_pretrained(
        self, path: str, *, sequence_lengths: Sequence[int]
    ) -> "GenMolMolecularGenerator":
        """Load GenMol V1 weights with matching constructor parameters.

        ``sequence_lengths`` supplies the token-length distribution used for generation.
        The checkpoint must contain tensors and plain Python values. Load saved
        torch-molecule estimators with ``load_from_local``.
        """
        lengths = list(sequence_lengths)
        if not lengths:
            raise ValueError("sequence_lengths must be nonempty.")
        try:
            checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                "Pretrained weights must contain tensors and plain Python values. "
                "Use a weights-only export and configure GenMol through its constructor parameters."
            ) from exc
        if "state_dict" not in checkpoint:
            raise ValueError("Expected a native GenMol checkpoint with a state_dict.")
        config = checkpoint.get("hyper_parameters", {}).get("config", {})
        if config.get("training", {}).get("use_bracket_safe", False):
            raise ValueError("This generator supports GenMol V1 checkpoints, not bracket-SAFE V2.")
        native_model = config.get("model", {})
        architecture = {
            "num_hidden_layers": self.num_layer,
            "num_attention_heads": self.num_head,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_len,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.dropout,
            "attention_probs_dropout_prob": self.attention_dropout,
            "layer_norm_eps": self.layer_norm_eps,
            "type_vocab_size": self.type_vocab_size,
            "position_embedding_type": "absolute",
            "vocab_size": 1880,
            "pad_token_id": 3,
        }
        mismatches = [
            name
            for name, value in architecture.items()
            if name in native_model and native_model[name] != value
        ]
        if mismatches:
            raise ValueError(f"Native checkpoint architecture does not match: {mismatches}.")
        previous = self.__dict__.copy()
        try:
            self.tokenizer_json_ = None
            self._initialize_model(self.model_class)
            state = {
                key[len("backbone.") :]: value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("backbone.")
            }
            self.model.backbone.load_state_dict(state, strict=True)
            if self.model.ema is not None:
                if "ema" not in checkpoint:
                    raise ValueError(
                        "Native checkpoint lacks EMA; set ema_decay=0 to load raw weights."
                    )
                self.model.set_extra_state({"ema": checkpoint["ema"]})
            self.sequence_lengths_ = lengths
            self.fitting_epoch = 0
            self.fitting_loss = []
            self.model.eval()
            self.is_fitted_ = True
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise
        return self
