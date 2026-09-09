from unittest.mock import MagicMock

import torch

from torch_molecule.generator.pretrained.families.seq2seq import (
    DEFAULT_MOLGEN_PREFIX_SELFIES,
    generate_seq2seq,
)


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        length = len(text)
        return {
            "input_ids": torch.tensor([[1, length, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return f"[SELFIES_{int(sequence[0].item())}]"


class _FakeModel:
    def generate(self, **kwargs):
        num_sequences = kwargs["num_return_sequences"]
        seq_len = kwargs.get("max_length", 8)
        return torch.arange(num_sequences * seq_len, dtype=torch.long).reshape(num_sequences, seq_len)


def test_generate_seq2seq_default_prefix():
    model = _FakeModel()
    model.generate = MagicMock(side_effect=_FakeModel().generate)

    outputs = generate_seq2seq(
        model,
        _FakeTokenizer(),
        torch.device("cpu"),
        n_samples=3,
        max_length=12,
        min_length=4,
        num_beams=5,
    )

    assert len(outputs) == 3
    call_kwargs = model.generate.call_args.kwargs
    assert call_kwargs["num_return_sequences"] == 3
    assert call_kwargs["num_beams"] == 5
    assert call_kwargs["max_length"] == 12


def test_generate_seq2seq_expands_beams_for_sample_count():
    model = MagicMock()
    model.generate.return_value = torch.zeros(7, 5, dtype=torch.long)

    generate_seq2seq(
        model,
        _FakeTokenizer(),
        torch.device("cpu"),
        n_samples=7,
        num_beams=3,
    )

    assert model.generate.call_args.kwargs["num_beams"] == 7


def test_default_molgen_prefix_is_benzene_selfies():
    assert "Ring1" in DEFAULT_MOLGEN_PREFIX_SELFIES
