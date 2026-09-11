import warnings
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ...base import BaseMolecularGenerator
from .checkpoint import build_metadata, load_metadata, save_metadata
from .families.causal_lm import generate_causal_lm
from .families.molexar import extract_conditions, generate_molexar
from .families.seq2seq import generate_seq2seq
from .finetune import finetune_generator
from .registry import (
    CAUSAL_LM_FAMILIES,
    MOLEXAR_FAMILIES,
    SAFE_GPT_FAMILIES,
    SEQ2SEQ_FAMILIES,
    resolve_family,
)


class HFPretrainedMolecularGenerator(BaseMolecularGenerator):
    """Hugging Face pretrained models as molecular generators.

    This class loads pretrained generative models from Hugging Face and exposes
    a sklearn-style ``fit`` / ``generate`` interface consistent with other
    generators in torch-molecule.

    Supported generation modes depend on the model family:

    - NovoMolGen: de novo SMILES generation from BOS.
    - MolGen-large / MolGen-large-opt: SELFIES seq2seq generation via ``prefix_selfies=``
      or ``scaffold=`` (SMILES converted internally).
    - Molexar: Fragment-SELFIES de novo and fragment-constrained generation via
      ``start_smiles`` / ``start_string`` / ``conditions`` (omni).
    - SAFE-GPT: GPT-2 causal LM on SAFE strings; de novo and ``scaffold=`` prefix.

    Other registered families can be loaded but may raise ``NotImplementedError``
    until later phases are implemented.

    Tested models include:

    - NovoMolGen: Causal LM pretrained on ZINC-22 for de novo SMILES generation.

      repo_id: ``"chandar-lab/NovoMolGen_32M_SMILES_BPE"``
      (https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_BPE)

    - MolGen-large: Seq2Seq SELFIES generator with high chemical validity.

      repo_id: ``"zjunlp/MolGen-large"``
      (https://huggingface.co/zjunlp/MolGen-large)

    - MolGen-large-opt: MolGen-large fine-tuned for QED / p-logP optimization.

      repo_id: ``"zjunlp/MolGen-large-opt"``
      (https://huggingface.co/zjunlp/MolGen-large-opt)

    - Molexar-10M-base: Fragment-SELFIES de novo and fragment-constrained generation.

      repo_id: ``"fairydance/molexar-10m-base"``
      (https://huggingface.co/fairydance/molexar-10m-base)

    - Molexar-10M-omni: Multi-condition Molexar model for property-guided generation.

      repo_id: ``"fairydance/molexar-10m-omni"``
      (https://huggingface.co/fairydance/molexar-10m-omni)

    - SAFE-GPT: GPT-2 causal LM pretrained on 1.1B SAFE strings for de novo
      generation and scaffold-prefix completion.

      repo_id: ``"datamol-io/safe-gpt"``
      (https://huggingface.co/datamol-io/safe-gpt)

    Parameters
    ----------
    repo_id : str
        Hugging Face repository id of the pretrained generator.
    max_length : int, default=128
        Maximum sequence length used when loading the tokenizer.
    revision : Optional[str], default=None
        Model revision on the Hugging Face Hub. NovoMolGen defaults to
        ``"hf-checkpoint"`` so standard ``model.generate`` works out of the box.
    trust_remote_code : bool, default=False
        Whether to trust remote code when loading from Hugging Face.
        Automatically enabled for Molexar.
    tokenizer_repo_id : Optional[str], default=None
        Optional Hugging Face repo for the tokenizer.
    generate_max_length : int, default=64
        Default ``max_length`` passed to ``generate()``.
    batch_size : int, default=8
        Batch size used when fine-tuning on SMILES data.
    epochs : int, default=1
        Number of fine-tuning epochs when ``fit(X)`` is called.
    learning_rate : float, default=5e-5
        Learning rate for fine-tuning.
    weight_decay : float, default=0.01
        Weight decay for fine-tuning.
    grad_norm_clip : Optional[float], default=1.0
        Maximum gradient norm during fine-tuning. Set to ``None`` to disable clipping.
    device : Optional[Union[torch.device, str]], default=None
        Device to run the model on.
    model_name : str, default="HFPretrainedMolecularGenerator"
        Name identifier for the model instance.
    verbose : str, default="none"
        Progress display mode: ``"none"``, ``"progress_bar"``, or
        ``"print_statement"``.
    """

    def __init__(
        self,
        repo_id: str,
        max_length: int = 128,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        tokenizer_repo_id: Optional[str] = None,
        generate_max_length: int = 64,
        batch_size: int = 8,
        epochs: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        grad_norm_clip: Optional[float] = 1.0,
        *,
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "HFPretrainedMolecularGenerator",
        verbose: str = "none",
    ):
        super().__init__(device=device, model_name=model_name, verbose=verbose)

        self.repo_id = repo_id
        self.max_length = max_length
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.tokenizer_repo_id = tokenizer_repo_id
        self.generate_max_length = generate_max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_norm_clip = grad_norm_clip
        self.fitting_loss: List[float] = []

        self._family: Optional[str] = None
        self.tokenizer = None
        self._molexar_engine = None
        self._model_local_path: Optional[str] = None
        self.fitting_epoch = -1

        self._require_transformers()

        if resolve_family(self.repo_id) == "causal_lm":
            warnings.warn(
                f"Unknown repo_id: {self.repo_id}. The class will try to load the "
                "model from Hugging Face as a causal LM, but generation may fail "
                "if the architecture is not supported.",
                stacklevel=2,
            )

    @staticmethod
    def _get_param_names() -> List[str]:
        return [
            "repo_id",
            "max_length",
            "revision",
            "trust_remote_code",
            "tokenizer_repo_id",
            "generate_max_length",
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "grad_norm_clip",
            "model_name",
        ]

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "max_length": self.max_length,
            "generate_max_length": self.generate_max_length,
        }

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer, None

    def _train_epoch(self, train_loader, optimizer) -> Dict[str, float]:
        raise NotImplementedError(
            "Use fit(X) for fine-tuning HFPretrainedMolecularGenerator."
        )

    def save_to_local(self, path: str) -> None:
        """Save the model and tokenizer to a local directory."""
        self._check_is_fitted()
        os.makedirs(path, exist_ok=True)

        if self._family in MOLEXAR_FAMILIES:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        save_metadata(
            path,
            build_metadata(
                repo_id=self.repo_id,
                family=self._family,
                max_length=self.max_length,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                tokenizer_repo_id=self.tokenizer_repo_id,
                generate_max_length=self.generate_max_length,
                model_name=self.model_name,
            ),
        )
        self._model_local_path = path

    def load_from_local(self, path: str) -> None:
        """Load a model and tokenizer saved by :meth:`save_to_local`."""
        metadata = load_metadata(path)
        self.repo_id = metadata["repo_id"]
        self._family = metadata["family"]
        self.max_length = metadata.get("max_length", self.max_length)
        self.revision = metadata.get("revision")
        self.trust_remote_code = metadata.get("trust_remote_code", self.trust_remote_code)
        self.tokenizer_repo_id = metadata.get("tokenizer_repo_id")
        self.generate_max_length = metadata.get("generate_max_length", self.generate_max_length)
        self.model_name = metadata.get("model_name", self.model_name)
        self._model_local_path = path

        if self._family in MOLEXAR_FAMILIES:
            self._load_molexar_pretrained(local_path=path)
        else:
            self._load_pretrained(local_path=path)

        self.is_fitted_ = True

    def save_to_hf(self, repo_id: str, **kwargs) -> None:
        raise NotImplementedError(
            "HFPretrainedMolecularGenerator does not support saving to Hugging Face."
        )

    def load_from_hf(self, repo_id: Optional[str] = None, **kwargs) -> None:
        """Load the pretrained model from Hugging Face (same as ``fit()``)."""
        if repo_id is not None:
            self.repo_id = repo_id
        self.fit()

    def load(self, path: Optional[str] = None, repo_id: Optional[str] = None, **kwargs) -> None:
        """Load the model from a local directory or Hugging Face."""
        if path is not None:
            self.load_from_local(path)
            return
        if repo_id is not None:
            self.repo_id = repo_id
        self.fit()

    def fit(
        self,
        X: Optional[List[str]] = None,
        y: Optional[np.ndarray] = None,
    ) -> "HFPretrainedMolecularGenerator":
        """Load the pretrained model from Hugging Face.

        Parameters
        ----------
        X : Optional[List[str]], default=None
            Optional SMILES strings for fine-tuning. When provided, the pretrained
            weights are adapted on the encoded family-specific representation.
        y : Optional[np.ndarray], default=None
            Reserved for future conditional fine-tuning. Currently ignored with a warning.

        Returns
        -------
        HFPretrainedMolecularGenerator
            The fitted generator instance.
        """
        assert self.repo_id is not None, "repo_id is not set"
        self._require_transformers()

        self._family = resolve_family(self.repo_id)
        self._load_pretrained()

        if X is not None:
            if y is not None:
                warnings.warn(
                    "Conditional fine-tuning with y is not implemented yet; continuing with "
                    "unconditional language-model fine-tuning.",
                    stacklevel=2,
                )
                y = None
            X, y = self._validate_inputs(X, y, return_rdkit_mol=False)
            X = self._encode_inputs(X)
            self._finetune(X, y)

        self.is_fitted_ = True
        return self

    def generate(self, n_samples: int = 10, **kwargs) -> List[str]:
        """Generate molecules as SMILES strings.

        Parameters
        ----------
        n_samples : int, default=10
            Number of molecules to generate.
        **kwargs
            Additional arguments forwarded to the family-specific generator.
            For causal LMs, common options include ``max_length``, ``temperature``,
            ``do_sample``, and ``scaffold``. For MolGen, use ``prefix_selfies`` or
            ``scaffold`` plus optional ``num_beams``, ``min_length``, and
            ``max_length``. For Molexar, use ``start_smiles``, ``start_string``,
            ``generation_task``, or ``conditions`` for omni models. For SAFE-GPT,
            use ``scaffold=`` with a SMILES prefix (converted to SAFE internally).

        Returns
        -------
        List[str]
            Generated SMILES strings. For MolGen, Molexar, and SAFE-GPT, invalid
            decodes are dropped, so the list may be shorter than ``n_samples``.
        """
        self._check_is_fitted()

        if self._family in CAUSAL_LM_FAMILIES:
            scaffold = kwargs.pop("scaffold", None)
            if scaffold is not None and self._family in SAFE_GPT_FAMILIES:
                from .utils import smiles_to_safe

                scaffold = smiles_to_safe([scaffold])[0]
            raw = generate_causal_lm(
                self.model,
                self.tokenizer,
                self.device,
                n_samples,
                family=self._family,
                max_length=kwargs.pop("max_length", self.generate_max_length),
                temperature=kwargs.pop("temperature", 1.0),
                do_sample=kwargs.pop("do_sample", True),
                scaffold=scaffold,
                **kwargs,
            )
            return self._decode_outputs(raw)

        if self._family in SEQ2SEQ_FAMILIES:
            raw = generate_seq2seq(
                self.model,
                self.tokenizer,
                self.device,
                n_samples,
                prefix_selfies=self._resolve_prefix_selfies(kwargs),
                max_length=kwargs.pop("max_length", self.generate_max_length),
                min_length=kwargs.pop("min_length", 5),
                num_beams=kwargs.pop("num_beams", 5),
                **kwargs,
            )
            return self._decode_outputs(raw)

        if self._family in MOLEXAR_FAMILIES:
            conditions = extract_conditions(kwargs)
            raw = generate_molexar(
                self._molexar_engine,
                n_samples,
                conditions=conditions or None,
                max_new_tokens=kwargs.pop("max_new_tokens", None),
                temperature=kwargs.pop("temperature", 0.8),
                top_p=kwargs.pop("top_p", 0.95),
                top_k=kwargs.pop("top_k", 50),
                do_sample=kwargs.pop("do_sample", True),
                repetition_penalty=kwargs.pop("repetition_penalty", 1.0),
                batch_size=kwargs.pop("batch_size", 100),
                start_string=kwargs.pop("start_string", None),
                start_smiles=kwargs.pop("start_smiles", None),
                start_fragment_selfies=kwargs.pop("start_fragment_selfies", None),
                generation_task=kwargs.pop("generation_task", None),
                **kwargs,
            )
            return self._decode_outputs(raw)

        raise NotImplementedError(f"Generation is not implemented for family '{self._family}'.")

    def _encode_inputs(self, smiles: List[str]) -> List[str]:
        """Convert SMILES inputs to the representation expected by the model family."""
        if self._family in SEQ2SEQ_FAMILIES:
            from .utils import smiles_to_selfies

            return smiles_to_selfies(smiles)
        if self._family in MOLEXAR_FAMILIES:
            from .utils import smiles_to_fragment_selfies

            return smiles_to_fragment_selfies(smiles)
        if self._family in SAFE_GPT_FAMILIES:
            from .utils import smiles_to_safe

            return smiles_to_safe(smiles)
        return smiles

    def _resolve_prefix_selfies(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Resolve a MolGen SELFIES prefix from kwargs."""
        prefix_selfies = kwargs.pop("prefix_selfies", None)
        scaffold = kwargs.pop("scaffold", None)

        if prefix_selfies is not None:
            return prefix_selfies
        if scaffold is not None:
            from .utils import smiles_to_selfies

            return smiles_to_selfies([scaffold])[0]
        return None

    def _decode_outputs(self, outputs: List[str]) -> List[str]:
        """Normalize raw model strings to SMILES.

        MolGen, Molexar, and SAFE-GPT drop strings that cannot be decoded. The
        returned list length is the number of successful SMILES, which may be
        smaller than ``n_samples``.
        """
        n_attempted = len(outputs)

        if self._family in SEQ2SEQ_FAMILIES:
            from .utils import selfies_to_smiles

            cleaned = [output.replace(" ", "") for output in outputs]
            smiles = selfies_to_smiles(cleaned)
        elif self._family in MOLEXAR_FAMILIES:
            from .utils import fragment_selfies_to_smiles

            smiles = fragment_selfies_to_smiles(outputs)
        elif self._family in SAFE_GPT_FAMILIES:
            from .utils import safe_to_smiles

            smiles = safe_to_smiles(outputs)
        else:
            return [output.replace(" ", "") for output in outputs]

        if len(smiles) < n_attempted:
            warnings.warn(
                f"got {len(smiles)}/{n_attempted} valid SMILES",
                stacklevel=2,
            )
        return smiles

    def _load_pretrained(self, local_path: Optional[str] = None) -> None:
        import transformers

        if self._family in MOLEXAR_FAMILIES:
            self._load_molexar_pretrained(local_path=local_path)
            return

        load_kwargs = self._get_load_kwargs()
        model_cls = self._get_model_class()
        model_source = local_path or self.repo_id
        tokenizer_repo = local_path or self.tokenizer_repo_id or self.repo_id

        if self._family in SAFE_GPT_FAMILIES:
            self.tokenizer = self._load_safe_gpt_tokenizer(tokenizer_repo)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_repo,
                model_max_length=self.max_length,
                **load_kwargs,
            )
        self.model = model_cls.from_pretrained(model_source, **load_kwargs)
        self._setup_tokenizer()
        self.model.to(self.device)
        self.model.eval()

    def _load_safe_gpt_tokenizer(self, tokenizer_repo: str):
        """Load the custom SAFE tokenizer as a Hugging Face fast tokenizer."""
        from .utils import _require_safe

        _require_safe()
        from safe.tokenizer import SAFETokenizer

        tokenizer_kwargs = {}
        if self.revision is not None:
            tokenizer_kwargs["revision"] = self.revision
        safe_tokenizer = SAFETokenizer.from_pretrained(tokenizer_repo, **tokenizer_kwargs)
        tokenizer = safe_tokenizer.get_pretrained()
        tokenizer.model_max_length = self.max_length
        return tokenizer

    def _load_molexar_pretrained(self, local_path: Optional[str] = None) -> None:
        from huggingface_hub import snapshot_download

        from .families.molexar import _require_molexar

        _require_molexar()
        from molexar.inference import MolexarInference

        if local_path is None:
            self._model_local_path = snapshot_download(self.repo_id)
        else:
            self._model_local_path = local_path

        device = str(self.device)
        self._molexar_engine = MolexarInference(
            self._model_local_path,
            device=device,
            tokenizer_path=self.tokenizer_repo_id,
        )
        self.model = self._molexar_engine.model
        self.tokenizer = self._molexar_engine.tokenizer

    def _get_molexar_config(self):
        if self._molexar_engine is not None:
            return self._molexar_engine.config
        return getattr(self.model, "config", None)

    def _finetune(self, X: List[str], y: Optional[np.ndarray]) -> None:
        if len(X) == 0:
            raise ValueError("Fine-tuning requires at least one training example.")

        config = self._get_molexar_config() if self._family in MOLEXAR_FAMILIES else None
        losses, last_epoch = finetune_generator(
            self._family,
            self.model,
            self.tokenizer,
            X,
            self.device,
            config=config,
            max_length=self.max_length,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_norm_clip=self.grad_norm_clip,
            verbose=self.verbose,
        )
        self.fitting_loss = losses
        self.fitting_epoch = last_epoch
        self.model.eval()

    def _get_model_class(self):
        import transformers

        if self._family in SEQ2SEQ_FAMILIES:
            return transformers.AutoModelForSeq2SeqLM
        if self._family in SAFE_GPT_FAMILIES:
            # Hub config lists SAFEDoubleHeadsModel; the LM head is standard GPT-2.
            return transformers.GPT2LMHeadModel
        return transformers.AutoModelForCausalLM

    def _get_load_kwargs(self) -> Dict[str, Any]:
        load_kwargs: Dict[str, Any] = {}

        if self._family == "novomolgen" and self.revision is None:
            load_kwargs["revision"] = "hf-checkpoint"
        elif self.revision is not None:
            load_kwargs["revision"] = self.revision

        if self._family in MOLEXAR_FAMILIES or self.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        if self._family in SAFE_GPT_FAMILIES:
            # Extra property-prediction head in the checkpoint is unused.
            load_kwargs["ignore_mismatched_sizes"] = True

        return load_kwargs

    def _setup_tokenizer(self) -> None:
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self._family in SAFE_GPT_FAMILIES:
            config = getattr(self.model, "config", None)
            if self.tokenizer.bos_token_id is None and getattr(config, "bos_token_id", None) is not None:
                self.tokenizer.bos_token_id = config.bos_token_id
            if self.tokenizer.eos_token_id is None and getattr(config, "eos_token_id", None) is not None:
                self.tokenizer.eos_token_id = config.eos_token_id
            if self.tokenizer.pad_token_id is None and getattr(config, "pad_token_id", None) is not None:
                self.tokenizer.pad_token_id = config.pad_token_id

    @staticmethod
    def _require_transformers() -> None:
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for HFPretrainedMolecularGenerator. "
                "Please install it using `pip install transformers`."
            ) from exc
