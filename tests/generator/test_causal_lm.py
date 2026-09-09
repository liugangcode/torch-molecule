from unittest.mock import MagicMock

import pytest
import torch

from torch_molecule.generator.pretrained.families.causal_lm import generate_causal_lm


class _FakeTokenizer:
    bos_token_id = 1
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        token_ids = [10 + len(text), 11 + len(text)]
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return {"input_ids": torch.tensor([token_ids])}

    def batch_decode(self, outputs, skip_special_tokens=True):
        return [f"SMILES_{idx}" for idx in range(outputs.shape[0])]


class _FakeModel:
    def generate(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0] if "input_ids" in kwargs else kwargs["num_return_sequences"]
        seq_len = kwargs.get("max_length", 8)
        return torch.zeros(batch_size, seq_len, dtype=torch.long)


def test_generate_causal_lm_bos_path():
    outputs = generate_causal_lm(
        _FakeModel(),
        _FakeTokenizer(),
        torch.device("cpu"),
        n_samples=3,
        family="novomolgen",
        max_length=12,
        do_sample=False,
    )
    assert outputs == ["SMILES_0", "SMILES_1", "SMILES_2"]


def test_generate_causal_lm_scaffold_path():
    model = _FakeModel()
    model.generate = MagicMock(return_value=torch.zeros(2, 8, dtype=torch.long))

    outputs = generate_causal_lm(
        model,
        _FakeTokenizer(),
        torch.device("cpu"),
        n_samples=2,
        family="gp_molformer",
        scaffold="c1ccccc1",
        max_length=12,
        do_sample=False,
    )
    assert outputs == ["SMILES_0", "SMILES_1"]
    kwargs = model.generate.call_args.kwargs
    assert kwargs["use_cache"] is False
    assert kwargs["top_k"] is None
    # Default tokenize is [BOS, ..., EOS]; IBM drops the trailing special token.
    assert kwargs["input_ids"].tolist() == [[1, 18, 19], [1, 18, 19]]


def test_generate_causal_lm_novomolgen_scaffold_keeps_all_tokens():
    model = _FakeModel()
    model.generate = MagicMock(return_value=torch.zeros(2, 8, dtype=torch.long))

    generate_causal_lm(
        model,
        _FakeTokenizer(),
        torch.device("cpu"),
        n_samples=2,
        family="novomolgen",
        scaffold="c1ccccc1",
        max_length=12,
        do_sample=False,
    )

    assert model.generate.call_args.kwargs["input_ids"].tolist() == [
        [18, 19],
        [18, 19],
    ]
    assert "top_k" not in model.generate.call_args.kwargs


def test_generate_causal_lm_gp_molformer_denovo_path():
    model = _FakeModel()
    model.generate = MagicMock(return_value=torch.zeros(2, 8, dtype=torch.long))

    outputs = generate_causal_lm(
        model,
        _FakeTokenizer(),
        torch.device("cpu"),
        n_samples=2,
        family="gp_molformer",
        max_length=12,
        do_sample=True,
    )

    assert outputs == ["SMILES_0", "SMILES_1"]
    assert model.generate.call_args.kwargs["num_return_sequences"] == 2
    assert model.generate.call_args.kwargs["use_cache"] is False
    assert "input_ids" not in model.generate.call_args.kwargs


def test_generate_causal_lm_novomolgen_does_not_force_use_cache_false():
    model = _FakeModel()
    model.generate = MagicMock(return_value=torch.zeros(3, 8, dtype=torch.long))

    generate_causal_lm(
        model,
        _FakeTokenizer(),
        torch.device("cpu"),
        n_samples=3,
        family="novomolgen",
        max_length=12,
        do_sample=False,
    )

    assert "use_cache" not in model.generate.call_args.kwargs
