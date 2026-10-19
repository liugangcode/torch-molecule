"""SAND molecular encoder."""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ...base import BaseMolecularEncoder
from .model import SANDModel

_MODEL_PARAMS = [
    "num_layer",
    "hidden_size",
    "emb_dim",
    "encoder_type",
    "num_atom_features",
    "num_edge_features",
    "normalize_shared_emb",
    "quantize_emb",
    "num_ivf_quantizers",
    "num_pq_quantizers",
    "ivf_codebook_size",
    "ivf_threshold_ema_dead_code",
    "ivf_ema_decay",
    "similarity_loss_weight",
    "regularization_strength",
    "uniformity_loss_weight",
    "uniformity_margin",
    "uniformity_num_matches",
    "ivf_loss_weight",
    "pq_loss_weight",
    "learning_rate",
    "min_lr",
    "weight_decay",
    "warmup_iters",
    "decay_iters",
]


class SANDMolecularEncoder(BaseMolecularEncoder):
    """SAND: Shape-Aware Neural Descriptor.

    Parameters
    ----------
    num_layer : int, default=7
        Number of message-passing layers.
    hidden_size : int, default=1024
        GNN hidden width.
    emb_dim : int, default=512
        Molecular embedding dimension.
    encoder_type : str, default="gine"
        GNN architecture: gine, gatv2, or gps.
    num_atom_features : int, default=62
        Atom feature vocabulary size; at least 62 for the fixed RDKit features.
    num_edge_features : int, default=16
        Edge feature vocabulary size; at least 16 for the fixed RDKit features.
    normalize_shared_emb : bool, default=True
        Normalize backbone embeddings during training and quantized reconstructions.
    quantize_emb : bool, default=True
        Enable IVF/PQ compression and both quantization losses. False selects dense training.
    num_ivf_quantizers : int, default=1
        Number of IVF quantizer heads.
    num_pq_quantizers : int, default=32
        Number of product-quantization subspaces.
    ivf_codebook_size : int, default=256
        Number of IVF centroids per head.
    ivf_threshold_ema_dead_code : float, default=0.5
        EMA count threshold for replacing unused IVF codes.
    ivf_ema_decay : float, default=0.8
        IVF centroid exponential-moving-average decay.
    similarity_loss_weight : float, default=1.0
        Weight of the soft-Spearman similarity loss.
    regularization_strength : float, default=1.0
        Temperature of the differentiable rank function.
    uniformity_loss_weight : float, default=0.5
        Weight of the embedding uniformity loss.
    uniformity_margin : float, default=0.2
        Cosine-similarity margin for the uniformity penalty.
    uniformity_num_matches : int, default=1
        Number of matches per query; must be 1.
    ivf_loss_weight : float, default=0.5
        Weight of the IVF cosine-alignment loss.
    pq_loss_weight : float, default=0.25
        Weight of the PQ squared-distance loss.
    batch_size : int, default=128
        Target pairs per training batch (at least 2), or molecules per encoding batch.
    epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=0.0005
        Peak learning rate.
    min_lr : float, default=0.00001
        Minimum rate after cosine decay.
    warmup_iters : int, default=10000
        Number of optimizer steps for learning-rate warmup.
    decay_iters : int, default=1000000
        Optimizer-step horizon for warmup and cosine decay.
    weight_decay : float, default=0.05
        AdamW weight decay.
    grad_clip_value : float, optional
        Maximum gradient norm. None disables clipping.
    emb_key : str, optional
        Embedding to return: emb or emb_quantized. None selects automatically.
    device : torch.device or str, optional
        CPU or CUDA device; defaults to CUDA when available.
    model_name : str, default="SANDMolecularEncoder"
        Estimator name.
    verbose : str, default="none"
        One of ``none``, ``progress_bar``, or ``print_statement``.

    References
    ----------
    Paper: https://proceedings.mlr.press/v306/winter26a.html
    Code: https://github.com/pfizer-opensource/SAND
    """

    def __init__(
        self,
        *,
        num_layer: int = 7,
        hidden_size: int = 1024,
        emb_dim: int = 512,
        encoder_type: str = "gine",
        num_atom_features: int = 62,
        num_edge_features: int = 16,
        normalize_shared_emb: bool = True,
        quantize_emb: bool = True,
        num_ivf_quantizers: int = 1,
        num_pq_quantizers: int = 32,
        ivf_codebook_size: int = 256,
        ivf_threshold_ema_dead_code: float = 0.5,
        ivf_ema_decay: float = 0.8,
        similarity_loss_weight: float = 1.0,
        regularization_strength: float = 1.0,
        uniformity_loss_weight: float = 0.5,
        uniformity_margin: float = 0.2,
        uniformity_num_matches: int = 1,
        ivf_loss_weight: float = 0.5,
        pq_loss_weight: float = 0.25,
        batch_size: int = 128,
        epochs: int = 100,
        learning_rate: float = 5e-4,
        min_lr: float = 1e-5,
        weight_decay: float = 0.05,
        warmup_iters: int = 10000,
        decay_iters: int = 1_000_000,
        grad_clip_value: Optional[float] = None,
        emb_key: Optional[str] = None,
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "SANDMolecularEncoder",
        verbose: str = "none",
    ):
        super().__init__(device=device, model_name=model_name, verbose=verbose)
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.encoder_type = encoder_type
        self.num_atom_features = num_atom_features
        self.num_edge_features = num_edge_features
        self.normalize_shared_emb = normalize_shared_emb
        self.quantize_emb = quantize_emb
        self.num_ivf_quantizers = num_ivf_quantizers
        self.num_pq_quantizers = num_pq_quantizers
        self.ivf_codebook_size = ivf_codebook_size
        self.ivf_threshold_ema_dead_code = ivf_threshold_ema_dead_code
        self.ivf_ema_decay = ivf_ema_decay
        self.similarity_loss_weight = similarity_loss_weight
        self.regularization_strength = regularization_strength
        self.uniformity_loss_weight = uniformity_loss_weight
        self.uniformity_margin = uniformity_margin
        self.uniformity_num_matches = uniformity_num_matches
        self.ivf_loss_weight = ivf_loss_weight
        self.pq_loss_weight = pq_loss_weight
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.batch_size = batch_size
        self.epochs = epochs
        self.grad_clip_value = grad_clip_value
        self.emb_key = emb_key
        self.fitting_loss = []
        self.fitting_epoch = 0
        self._global_step = 0
        self.model_class = SANDModel
        self._validate_hyperparameters()

    @staticmethod
    def _get_param_names() -> List[str]:
        return _MODEL_PARAMS + [
            "batch_size",
            "epochs",
            "grad_clip_value",
            "emb_key",
            "device",
            "model_name",
            "verbose",
            "fitting_loss",
            "fitting_epoch",
        ]

    def set_params(self, **params) -> "SANDMolecularEncoder":
        """Update settings, invalidating fitted weights when model settings change."""
        previous = self.__dict__.copy()
        try:
            if "device" in params:
                device = params["device"]
                params["device"] = (
                    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    if device is None
                    else torch.device(device)
                )
            super().set_params(**params)
            if any(
                name in _MODEL_PARAMS and getattr(self, name) != previous[name] for name in params
            ):
                self.is_fitted_ = False
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise
        return self

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        if checkpoint is None:
            params = {name: getattr(self, name) for name in _MODEL_PARAMS}
        else:
            params = {name: checkpoint["hyperparameters"][name] for name in _MODEL_PARAMS}
        return params

    def load_from_local(self, path: str) -> None:
        """Load a checkpoint, preserving the current encoder if loading fails."""
        previous = self.__dict__.copy()
        try:
            super().load_from_local(path)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

    def _validate_hyperparameters(self):
        for name in (
            "batch_size",
            "num_layer",
            "hidden_size",
            "emb_dim",
            "num_ivf_quantizers",
            "num_pq_quantizers",
            "ivf_codebook_size",
            "epochs",
            "decay_iters",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name, minimum in (("num_atom_features", 62), ("num_edge_features", 16)):
            if getattr(self, name) < minimum:
                raise ValueError(f"{name} must be at least {minimum} for the fixed RDKit features.")
        if self.encoder_type not in ("gine", "gatv2", "gps"):
            raise ValueError("encoder_type must be gine, gatv2, or gps.")
        if self.encoder_type != "gine" and self.hidden_size % 4:
            raise ValueError("hidden_size must be divisible by 4 for gatv2/gps attention heads.")
        if self.quantize_emb and (
            self.emb_dim % self.num_ivf_quantizers or self.emb_dim % self.num_pq_quantizers
        ):
            raise ValueError(
                "emb_dim must be divisible by num_ivf_quantizers and num_pq_quantizers."
            )
        if self.uniformity_num_matches != 1:
            raise ValueError("uniformity_num_matches must be 1 for query/match pairs.")
        if not 0 <= self.warmup_iters < self.decay_iters:
            raise ValueError("warmup_iters must be in [0, decay_iters).")
        if self.learning_rate <= 0 or self.regularization_strength <= 0:
            raise ValueError("learning_rate and regularization_strength must be positive.")
        if self.grad_clip_value is not None and self.grad_clip_value <= 0:
            raise ValueError("grad_clip_value must be positive or None.")
        if not 0 <= self.ivf_ema_decay < 1:
            raise ValueError("ivf_ema_decay must be in [0, 1).")
        weights = [self.similarity_loss_weight, self.uniformity_loss_weight]
        if self.quantize_emb:
            weights += [self.ivf_loss_weight, self.pq_loss_weight]
        if not any(weights):
            raise ValueError("At least one active loss weight must be positive.")
        if self.emb_key not in (None, "emb", "emb_quantized"):
            raise ValueError("emb_key must be None, emb, or emb_quantized.")
        if self.emb_key == "emb_quantized" and not self.quantize_emb:
            raise ValueError("emb_quantized requires quantize_emb=True.")

    def fit(self, X_train: List[str], y_train=None, *, pairs=None) -> "SANDMolecularEncoder":
        """Fit on molecular-pair similarity scores.

        Parameters
        ----------
        X_train : list of str
            Training SMILES strings.
        y_train : array-like of shape (n_pairs,)
            Similarity score for each pair.
        pairs : array-like of shape (n_pairs, 2)
            Query and match indices into X_train.

        Returns
        -------
        self : SANDMolecularEncoder
            Fitted encoder.
        """
        self._validate_hyperparameters()
        if self.batch_size < 2:
            raise ValueError("SAND training batch_size must be at least 2 pairs.")
        X_train, _ = self._validate_inputs(X_train, return_rdkit_mol=False)
        if not X_train:
            raise ValueError("X_train must contain at least one SMILES string.")
        if pairs is None or y_train is None:
            raise ValueError("SAND training requires pairs and y_train similarity scores.")
        pair_indices = np.asarray(pairs)
        if (
            pair_indices.ndim != 2
            or pair_indices.shape[1] != 2
            or pair_indices.dtype.kind not in "iu"
            or len(pair_indices) < 2
        ):
            raise ValueError("pairs must be an integer array of shape (M, 2), with M >= 2.")
        if np.any(pair_indices < 0) or np.any(pair_indices >= len(X_train)):
            raise ValueError("pairs indices must be within X_train.")
        scores = np.asarray(y_train, dtype=np.float32)
        if scores.ndim != 1 or len(scores) != len(pair_indices) or not np.isfinite(scores).all():
            raise ValueError("y_train must be a finite 1D score array matching pairs.")
        if np.all(scores == scores[0]):
            raise ValueError(
                "y_train must contain at least two different scores for native rank loss."
            )

        from torch.utils.data import DataLoader

        from .dataloader import PairBatchSampler, SANDMolecularGraphDataset
        from .mol_utils.rdkit.molecule import RDMolecule

        self.is_fitted_ = False
        self.fitting_loss = []
        self.fitting_epoch = 0
        self._global_step = 0
        self._initialize_model(self.model_class)
        graphs = [RDMolecule.from_smiles(smiles).to_graph() for smiles in X_train]
        dataset = SANDMolecularGraphDataset(graphs, pair_indices[:, 0], pair_indices[:, 1], scores)
        loader = DataLoader(
            dataset,
            batch_sampler=PairBatchSampler(scores, self.batch_size),
            collate_fn=dataset.collate_fn,
            num_workers=0,
        )
        optimizer, scheduler = self._setup_optimizers()
        progress = (
            tqdm(total=self.epochs * len(loader), desc="Training Progress")
            if self.verbose == "progress_bar"
            else None
        )
        try:
            for epoch in range(self.epochs):
                losses = self._train_epoch(loader, optimizer, epoch, progress)
                self.fitting_loss.append(float(np.mean(losses)))
                self.fitting_epoch = epoch
                if scheduler is not None:
                    scheduler.step(self.fitting_loss[-1])
        finally:
            if progress is not None:
                progress.close()
        self.model.eval()
        self.is_fitted_ = True
        return self

    def _setup_optimizers(self):
        """Configure AdamW."""
        return self.model.configure_optimizers(), None

    def _train_epoch(self, train_loader, optimizer, epoch=0, global_pbar=None):
        """Train one epoch."""
        self.model.train()
        losses = []
        for step, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Native SAND loss is non-finite; check for constant predicted "
                    "pair similarities or degenerate training molecules."
                )
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()
            # Preserve post-step scheduling: the first update uses the peak rate.
            self.model.update_lr(optimizer, self._global_step)
            self._global_step += 1
            losses.append(loss.item())
            log = {
                "Epoch": f"{epoch + 1}/{self.epochs}",
                "Step": f"{step + 1}/{len(train_loader)}",
                "Loss": f"{loss.item():.4f}",
            }
            if global_pbar is not None:
                global_pbar.set_postfix(log)
                global_pbar.update(1)
            if self.verbose == "print_statement":
                print(log)
        return losses

    def load_from_pretrained(self, checkpoint_path: str) -> "SANDMolecularEncoder":
        """Load pretrained SAND weights.

        Parameters
        ----------
        checkpoint_path : str
            Checkpoint file matching the model architecture.

        Returns
        -------
        self : SANDMolecularEncoder
            Encoder with loaded weights.
        """
        self._validate_hyperparameters()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.is_fitted_ = False
        self.fitting_loss = []
        self.fitting_epoch = 0
        self._global_step = 0
        self._initialize_model(self.model_class)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.is_fitted_ = True
        return self

    def encode(
        self, X: List[str], return_type: Literal["np", "pt"] = "pt"
    ) -> Union[np.ndarray, torch.Tensor]:
        """Return molecular embeddings as CPU tensors or NumPy arrays."""
        self._check_is_fitted()
        if return_type not in ("np", "pt"):
            raise ValueError("return_type must be 'np' or 'pt'.")
        X, _ = self._validate_inputs(X, return_rdkit_mol=False)
        if not X:
            raise ValueError("X must contain at least one SMILES string.")
        self.model.to(self.device).eval()
        embeddings = (
            self.model.encode_smiles(
                X, emb_key=self.emb_key, batch_size=self.batch_size, num_workers=0
            )
            .detach()
            .cpu()
        )
        return embeddings.numpy() if return_type == "np" else embeddings
