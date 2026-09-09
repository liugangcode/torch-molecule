"""Causal language model generation for SMILES-based HF generators."""

from typing import Any, List, Optional

import torch


def generate_causal_lm(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    n_samples: int,
    *,
    family: Optional[str] = None,
    max_length: int = 64,
    temperature: float = 1.0,
    do_sample: bool = True,
    scaffold: Optional[str] = None,
    **kwargs: Any,
) -> List[str]:
    """Generate SMILES strings with a causal language model.

    Parameters
    ----------
    model : torch.nn.Module
        A Hugging Face causal LM.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with the model.
    device : torch.device
        Device used for generation.
    n_samples : int
        Number of molecules to generate.
    family : Optional[str], default=None
        Generator family name. GP-MoLFormer uses a model-specific de novo path.
    max_length : int, default=64
        Maximum generated sequence length passed to ``model.generate``.
    temperature : float, default=1.0
        Sampling temperature.
    do_sample : bool, default=True
        Whether to use sampling during generation.
    scaffold : Optional[str], default=None
        Optional SMILES prefix for scaffold completion. For GP-MoLFormer this
        should be a *partial* SMILES string (IBM's example is ``c1cccc``);
        the official tokenizer appends a trailing special token that is then
        dropped so generation continues the prefix.

    Returns
    -------
    List[str]
        Raw decoded strings from the tokenizer (may contain spaces).
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    generate_kwargs = {
        "max_length": max_length,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    if family == "gp_molformer":
        # IBM remote code indexes tuple KV caches; transformers 4.56 injects
        # an empty DynamicCache that crashes prepare_inputs_for_generation.
        generate_kwargs["use_cache"] = False
        generate_kwargs["top_k"] = None

    generate_kwargs.update(kwargs)

    if scaffold:
        if family == "gp_molformer":
            # Match IBM/gp-molformer scripts/conditional_generation.py:
            # tokenize with special tokens, then drop the trailing SEP/EOS.
            input_ids = tokenizer(scaffold, return_tensors="pt")["input_ids"]
            if input_ids.shape[1] > 1:
                input_ids = input_ids[:, :-1]
        else:
            encoded = tokenizer(scaffold, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"]
        input_ids = input_ids.to(device).expand(n_samples, -1).contiguous()
        generate_kwargs["input_ids"] = input_ids
    elif family == "gp_molformer":
        generate_kwargs["num_return_sequences"] = n_samples
    else:
        if tokenizer.bos_token_id is None:
            raise ValueError(
                "Tokenizer has no BOS token. Provide `scaffold=` or use a model "
                "with a defined bos_token_id."
            )
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        generate_kwargs["input_ids"] = input_ids.expand(n_samples, -1).contiguous()

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
