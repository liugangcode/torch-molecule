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
        Generator family name. Unused by the standard ``generate()`` path;
        kept for call-site compatibility.
    max_length : int, default=64
        Maximum generated sequence length passed to ``model.generate``.
    temperature : float, default=1.0
        Sampling temperature.
    do_sample : bool, default=True
        Whether to use sampling during generation.
    scaffold : Optional[str], default=None
        Optional tokenized prefix. Callers that need a SMILES-to-SAFE
        conversion (SAFE-GPT) should pass the already-encoded prefix.

    Returns
    -------
    List[str]
        Raw decoded strings from the tokenizer (may contain spaces).
    """
    del family  # dispatch is done by HFPretrainedMolecularGenerator
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
    generate_kwargs.update(kwargs)

    if scaffold:
        encoded = tokenizer(scaffold, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        input_ids = input_ids.to(device).expand(n_samples, -1).contiguous()
        generate_kwargs["input_ids"] = input_ids
        generate_kwargs["attention_mask"] = torch.ones_like(input_ids)
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
