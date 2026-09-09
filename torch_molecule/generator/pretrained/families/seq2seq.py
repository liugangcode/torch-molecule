"""Seq2Seq generation for SELFIES-based HF generators such as MolGen."""

from typing import Any, List, Optional

import torch

DEFAULT_MOLGEN_PREFIX_SELFIES = "[C][=C][C][=C][C][=C][Ring1][=Branch1]"


def generate_seq2seq(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    n_samples: int,
    *,
    prefix_selfies: Optional[str] = None,
    max_length: int = 15,
    min_length: int = 5,
    num_beams: int = 5,
    **kwargs: Any,
) -> List[str]:
    """Generate SELFIES strings with a seq2seq language model.

    MolGen uses a corrupted SELFIES prefix as input and generates a completed
    SELFIES sequence via beam search.

    Parameters
    ----------
    model : torch.nn.Module
        A Hugging Face seq2seq model.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with the model.
    device : torch.device
        Device used for generation.
    n_samples : int
        Number of molecules to generate.
    prefix_selfies : Optional[str], default=None
        SELFIES prefix used as model input. Defaults to a benzene ring fragment.
    max_length : int, default=15
        Maximum generated sequence length.
    min_length : int, default=5
        Minimum generated sequence length.
    num_beams : int, default=5
        Beam width for beam search.

    Returns
    -------
    List[str]
        Raw decoded SELFIES strings from the tokenizer.
    """
    prefix = prefix_selfies or DEFAULT_MOLGEN_PREFIX_SELFIES
    encoded = tokenizer(prefix, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    beam_width = max(num_beams, n_samples)
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_length": max_length,
        "min_length": min_length,
        "num_return_sequences": n_samples,
        "num_beams": beam_width,
    }
    generate_kwargs.update(kwargs)

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    return [
        tokenizer.decode(
            sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        for sequence in outputs
    ]
