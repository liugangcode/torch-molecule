"""DemoDiff molecular generation with optional demonstration molecules."""

import math
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from ...base import BaseMolecularGenerator
from .data import DemoDiffTokenizer, ContextDataset, compute_dataset_info, context_collate_fn
from .model import DemoDiffModel


_MODEL_PARAMETERS = (
    "num_layer",
    "hidden_size",
    "num_head",
    "mlp_ratio",
    "diffusion_steps",
    "diffusion_noise_schedule",
    "drop_context",
    "no_context_train",
    "noise_weight",
    "lambda_train",
)


class DemoDiffMolecularGenerator(BaseMolecularGenerator):
    """Train a fragment-graph diffusion model, optionally using demonstrations.

    Parameters
    ----------
    num_layer : int, default=24
        Number of transformer layers.
    hidden_size : int, default=1280
        Transformer hidden dimension.
    num_head : int, default=16
        Attention heads; each head must have an even dimension.
    mlp_ratio : float, default=4.0
        Feed-forward expansion ratio.
    diffusion_steps : int, default=500
        Number of discrete diffusion steps.
    diffusion_noise_schedule : str, default="cosine"
        Noise schedule: "cosine", "linear", or "custom".
    context_length : int, default=150
        Maximum total fragment tokens in a target and its demonstrations.
    drop_context : float, default=0.1
        Probability of ignoring all demonstrations for a training batch.
    drop_incontext : float, default=0.0
        Initial drop probability for individual demonstrations, decreasing with order.
    no_context_train : bool, default=False
        Train without conditioning even when demonstrations are supplied.
    lambda_train : list of float, optional
        Token, bond, and attachment-position loss weights; defaults to [1, 1, 1].
    noise_weight : str or None, default=None
        Optional loss weighting: "noiser_lower" or "noiser_higher".
    guidance_scale : float, default=2.0
        Demonstration guidance strength during generation.
    batch_size : int, default=64
        Number of target molecules per batch.
    epochs : int, default=200
        Training epochs.
    learning_rate : float, default=2e-5
        AdamW learning rate.
    grad_clip_value : float or None, default=0.1
        Maximum gradient norm; None disables clipping.
    weight_decay : float, default=1e-12
        AdamW weight decay.
    verbose : str, default="none"
        "none", "progress_bar", or "print_statement".
    device : str or torch.device, optional
        Training and generation device; selected automatically when omitted.
    model_name : str, default="DemoDiffMolecularGenerator"
        Model identifier.

    Notes
    -----
    Uses the supplied 3,000-fragment vocabulary. The original tokenizer does not
    preserve all stereochemistry or formal charges. Polymer attachment atoms
    are supported. Demonstrations guide generation but are not fixed substructures.
    """

    def __init__(
        self,
        num_layer: int = 24,
        hidden_size: int = 1280,
        num_head: int = 16,
        mlp_ratio: float = 4.0,
        diffusion_steps: int = 500,
        diffusion_noise_schedule: str = "cosine",
        context_length: int = 150,
        drop_context: float = 0.1,
        drop_incontext: float = 0.0,
        no_context_train: bool = False,
        lambda_train: Optional[List[float]] = None,
        noise_weight: Optional[str] = None,
        guidance_scale: float = 2.0,
        batch_size: int = 64,
        epochs: int = 200,
        learning_rate: float = 2e-5,
        grad_clip_value: Optional[float] = 0.1,
        weight_decay: float = 1e-12,
        verbose: str = "none",
        *,
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "DemoDiffMolecularGenerator",
    ):
        super().__init__(device=device, model_name=model_name, verbose=verbose)
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.mlp_ratio = mlp_ratio
        self.diffusion_steps = diffusion_steps
        self.diffusion_noise_schedule = diffusion_noise_schedule
        self.context_length = context_length
        self.drop_context = drop_context
        self.drop_incontext = drop_incontext
        self.no_context_train = no_context_train
        self.lambda_train = list(lambda_train) if lambda_train is not None else [1.0, 1.0, 1.0]
        self.noise_weight = noise_weight
        self.guidance_scale = guidance_scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.grad_clip_value = grad_clip_value
        self.weight_decay = weight_decay
        self.dataset_info_ = None
        self.tokenizer = None
        self.fitting_loss = []
        self.fitting_epoch = 0
        self.model_class = DemoDiffModel
        self._validate_hyperparameters()

    @staticmethod
    def _get_param_names() -> List[str]:
        return list(_MODEL_PARAMETERS) + [
            "context_length",
            "drop_incontext",
            "guidance_scale",
            "batch_size",
            "epochs",
            "learning_rate",
            "grad_clip_value",
            "weight_decay",
            "dataset_info_",
            "fitting_loss",
            "fitting_epoch",
            "verbose",
            "device",
            "model_name",
        ]

    def _validate_hyperparameters(self):
        for name in (
            "num_layer",
            "hidden_size",
            "num_head",
            "diffusion_steps",
            "context_length",
            "batch_size",
            "epochs",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.hidden_size % (2 * self.num_head):
            raise ValueError("hidden_size must be divisible by 2 * num_head for rotary attention.")
        if self.mlp_ratio <= 0 or self.learning_rate <= 0:
            raise ValueError("mlp_ratio and learning_rate must be positive.")
        if self.grad_clip_value is not None and self.grad_clip_value <= 0:
            raise ValueError("grad_clip_value must be positive or None.")
        if not 0 <= self.drop_context <= 1 or not 0 <= self.drop_incontext <= 1:
            raise ValueError("Context drop probabilities must be in [0, 1].")
        if len(self.lambda_train) != 3 or min(self.lambda_train) < 0 or max(self.lambda_train) == 0:
            raise ValueError(
                "lambda_train must contain three nonnegative weights, with at least one positive."
            )
        if self.noise_weight not in (None, "noiser_lower", "noiser_higher"):
            raise ValueError("noise_weight must be None, 'noiser_lower', or 'noiser_higher'.")
        if self.diffusion_noise_schedule not in ("cosine", "linear", "custom"):
            raise ValueError("Unknown diffusion_noise_schedule.")
        if self.diffusion_noise_schedule == "custom" and self.diffusion_steps < 100:
            raise ValueError("The custom noise schedule requires at least 100 diffusion steps.")

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = self.get_params() if checkpoint is None else checkpoint["hyperparameters"]
        return {
            **{name: params[name] for name in _MODEL_PARAMETERS},
            "dataset_info": params["dataset_info_"],
            "guide_scale": params["guidance_scale"],
        }

    def _initialize_model(self, model_class, checkpoint=None):
        self.tokenizer = DemoDiffTokenizer()
        return super()._initialize_model(model_class, checkpoint)

    def _convert_to_pytorch_data(self, X: List[str], y=None) -> List[Data]:
        self.tokenizer = DemoDiffTokenizer()
        graphs = [self.tokenizer.encode(smiles) for smiles in X]
        if any(graph.num_nodes > self.context_length for graph in graphs):
            raise ValueError(
                "A molecule exceeds context_length fragment tokens; increase context_length."
            )
        return graphs

    def _setup_optimizers(self):
        return (
            torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=True,
                eps=1e-6,
            ),
            None,
        )

    def fit(self, X_train: List[str], y_train=None, *, contexts=None, score=None):
        """Train fresh weights on SMILES, optionally with related demonstrations.

        ``contexts[i]`` lists indices of other molecules in ``X_train`` shown
        alongside target ``i``. ``score[i]`` contains one finite similarity or
        normalized task score in [0, 1] per demonstration; higher means closer
        to the target. The target's own score is fixed to 1 internally. For
        structural similarity, supply Tanimoto similarity directly. Scores are
        used unchanged; use the same definition and scale in ``generate()``.
        With both arguments omitted, training is unconditional. ``y_train`` is
        unused; property labels are not converted into demonstrations automatically.
        """
        self._validate_hyperparameters()
        if y_train is not None:
            raise ValueError(
                "Use contexts and score for demonstration supervision; y_train is unsupported."
            )
        X_train, _ = self._validate_inputs(X_train, return_rdkit_mol=False)
        if not X_train:
            raise ValueError("X_train must contain at least one molecule.")
        graphs = self._convert_to_pytorch_data(X_train)
        if (contexts is None) != (score is None):
            raise ValueError("Provide both contexts and score, or neither.")
        if contexts is not None:
            if len(contexts) != len(graphs) or len(score) != len(graphs):
                raise ValueError("contexts and score must have one list per training molecule.")
            for target, (indices, values) in enumerate(zip(contexts, score)):
                values = np.asarray(values)
                if (
                    values.ndim != 1
                    or len(indices) != len(values)
                    or not np.isfinite(values).all()
                    or ((values < 0) | (values > 1)).any()
                ):
                    raise ValueError("Each context needs one finite score in [0, 1].")
                if any(
                    not isinstance(index, (int, np.integer))
                    or not 0 <= index < len(graphs)
                    or index == target
                    for index in indices
                ):
                    raise ValueError("Context indices must refer to other training molecules.")
                if (
                    graphs[target].num_nodes + sum(graphs[index].num_nodes for index in indices)
                    > self.context_length
                ):
                    raise ValueError(
                        "A target and its demonstrations exceed context_length fragment tokens."
                    )
        self.dataset_info_ = compute_dataset_info(graphs)
        train_dataset = ContextDataset(graphs, contexts, score, drop_incontext=self.drop_incontext)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=context_collate_fn,
        )
        self.is_fitted_ = False
        self._initialize_model(self.model_class)
        optimizer, _ = self._setup_optimizers()
        self.fitting_loss = []
        self.fitting_epoch = 0
        progress = (
            tqdm(total=self.epochs * len(train_loader), desc="Training Progress")
            if self.verbose == "progress_bar"
            else None
        )
        try:
            for epoch in range(self.epochs):
                losses = self._train_epoch(train_loader, optimizer, epoch, progress)
                self.fitting_loss.append(float(np.mean(losses)))
                self.fitting_epoch = epoch
        finally:
            if progress is not None:
                progress.close()
        self.model.eval()
        self.is_fitted_ = True
        return self

    def _train_epoch(self, train_loader, optimizer, epoch=0, global_pbar=None):
        self.model.train()
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = self.model.compute_loss(batch)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()
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

    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 10,
        *,
        contexts: Optional[List[str]] = None,
        score: Optional[List[float]] = None,
        num_nodes: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        max_attempts: int = 10,
    ) -> List[str]:
        """Generate SMILES, optionally guided by shared demonstration molecules.

        ``contexts`` contains demonstration SMILES. ``score`` supplies one finite
        similarity or normalized task score in [0, 1] per demonstration, using
        the same definition and scale as in ``fit()``. Higher means closer to
        the target, whose query score is fixed to 1 internally. Scores describe
        the demonstrations and guide generation; they do not guarantee measured
        similarities or properties of the generated molecules.

        ``num_nodes`` fixes the number of fragment tokens;
        otherwise their count is sampled from training data. Only valid, connected
        decoded molecules are returned. Exhausting ``max_attempts`` per batch
        warns and returns available results, which may include duplicates.
        """
        if not self.is_fitted_ or self.model is None:
            raise ValueError("DemoDiff must be fitted or loaded before generation.")
        if n_samples < 0 or max_attempts < 1:
            raise ValueError("n_samples must be nonnegative and max_attempts positive.")
        if n_samples == 0:
            return []
        if (contexts is None) != (score is None):
            raise ValueError("Provide both contexts and score, or neither.")
        context_graphs = []
        if contexts is not None:
            contexts, _ = self._validate_inputs(contexts, return_rdkit_mol=False)
            values = np.asarray(score)
            if (
                values.ndim != 1
                or len(contexts) != len(values)
                or not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()
            ):
                raise ValueError("Each context needs one finite score in [0, 1].")
            context_graphs = [self.tokenizer.encode(smiles) for smiles in contexts]
        capacity = self.dataset_info_["max_n_nodes"]
        if any(graph.num_nodes > capacity for graph in context_graphs):
            raise ValueError("A demonstration exceeds the trained fragment capacity.")
        if num_nodes is not None and not 1 <= num_nodes <= capacity:
            raise ValueError(f"num_nodes must be between 1 and {capacity}.")
        available = self.context_length - sum(graph.num_nodes for graph in context_graphs)
        if available < 1:
            raise ValueError("Demonstrations leave no context_length for a generated molecule.")
        counts = torch.tensor(self.dataset_info_["node_counts"], dtype=torch.float)
        if num_nodes is not None:
            if num_nodes > available:
                raise ValueError("The requested molecule and demonstrations exceed context_length.")
        elif available < capacity and counts[available + 1 :].sum() > 0:
            raise ValueError(
                "Demonstrations leave insufficient context_length for trained molecule sizes."
            )
        was_training = self.model.training
        previous_scale = self.model.guide_scale
        self.model.eval()
        self.model.guide_scale = self.guidance_scale if guidance_scale is None else guidance_scale
        generated = []
        try:
            for _ in range(max_attempts * math.ceil(n_samples / self.batch_size)):
                count = min(self.batch_size, n_samples - len(generated))
                sizes = (
                    [num_nodes] * count
                    if num_nodes is not None
                    else torch.multinomial(counts, count, replacement=True).tolist()
                )
                batch = []
                for size in sizes:
                    target = Data(
                        x=torch.zeros(size, dtype=torch.long),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_attr=torch.empty(0, dtype=torch.long),
                        edge_pos=torch.empty(0, dtype=torch.long),
                        y=torch.empty((1, 0)),
                    )
                    batch.append(
                        {
                            "target": 0,
                            "sequence": list(range(1 + len(context_graphs))),
                            "score": [1.0] + (list(score) if score is not None else []),
                            "molecular_graphs": [target] + context_graphs,
                        }
                    )
                samples, _ = self.model.sample_batch(
                    context_collate_fn(batch), init_node="reconstruction"
                )
                for nodes, bonds, positions in samples:
                    smiles = self.tokenizer.decode(nodes, bonds, positions)
                    if smiles is not None:
                        molecule = Chem.MolFromSmiles(smiles)
                        if (
                            molecule is not None
                            and molecule.GetNumAtoms()
                            and len(Chem.GetMolFrags(molecule)) == 1
                        ):
                            generated.append(Chem.MolToSmiles(molecule))
                if len(generated) == n_samples:
                    break
        finally:
            self.model.train(was_training)
            self.model.guide_scale = previous_scale
        if len(generated) < n_samples:
            warnings.warn(
                f"DemoDiff generated {len(generated)} of {n_samples} valid molecules; returning available results.",
                UserWarning,
                stacklevel=2,
            )
        return generated
