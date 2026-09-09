import json
import os

import pytest
import torch

from torch_molecule.generator.pretrained.checkpoint import (
    METADATA_FILENAME,
    build_metadata,
    load_metadata,
    save_metadata,
)
from torch_molecule.generator.pretrained.finetune import (
    corrupt_token_ids,
    finetune_causal_lm,
    finetune_generator,
    finetune_seq2seq,
)


class _FakeOutput:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss


class _FakeLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        return _FakeOutput(self.weight * 0.0 + 1.0)


def test_finetune_causal_lm_runs_one_epoch():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    losses, last_epoch = finetune_causal_lm(
        _FakeLM(),
        tokenizer,
        ["CCO", "CC(=O)O"],
        torch.device("cpu"),
        max_length=16,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_norm_clip=1.0,
        verbose="none",
    )
    assert last_epoch == 0
    assert len(losses) == 1
    assert losses[0] == pytest.approx(1.0)


class _FakeTokenizer:
    mask_token_id = 99
    pad_token_id = 0
    all_special_ids = [0, 1, 2]


def _seq2seq_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    return tokenizer


def test_corrupt_token_ids_masks_non_special_tokens():
    input_ids = torch.tensor([1, 10, 11, 12, 2])
    generator = torch.Generator().manual_seed(0)
    corrupted = corrupt_token_ids(
        input_ids,
        _FakeTokenizer(),
        mask_prob=1.0,
        generator=generator,
    )
    assert corrupted[0].item() == 1
    assert corrupted[-1].item() == 2
    assert torch.equal(corrupted[1:-1], torch.tensor([99, 99, 99]))


def test_corrupt_token_ids_requires_mask_token():
    class _NoMask:
        mask_token_id = None

    with pytest.raises(ValueError, match="mask_token_id"):
        corrupt_token_ids(torch.tensor([1, 2, 3]), _NoMask())


def test_finetune_seq2seq_runs_one_epoch():
    pytest.importorskip("transformers")

    tokenizer = _seq2seq_tokenizer()
    losses, last_epoch = finetune_seq2seq(
        _FakeLM(),
        tokenizer,
        ["[C][C][O]", "[C][C][Branch1][C][O]"],
        torch.device("cpu"),
        max_length=16,
        batch_size=1,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_norm_clip=None,
        verbose="none",
        mask_prob=1.0,
    )
    assert last_epoch == 0
    assert len(losses) == 1


def test_finetune_seq2seq_labels_stay_clean():
    pytest.importorskip("transformers")

    tokenizer = _seq2seq_tokenizer()
    captured = {}

    class _CaptureLM(_FakeLM):
        def forward(self, **kwargs):
            captured["input_ids"] = kwargs["input_ids"].detach().clone()
            captured["labels"] = kwargs["labels"].detach().clone()
            return super().forward(**kwargs)

    finetune_seq2seq(
        _CaptureLM(),
        tokenizer,
        ["[C][C][O][C][C][O]"],
        torch.device("cpu"),
        max_length=16,
        batch_size=1,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_norm_clip=None,
        verbose="none",
        mask_prob=1.0,
    )

    labels = captured["labels"]
    input_ids = captured["input_ids"]
    ignore_index = -100
    content = labels[0] != ignore_index
    assert not torch.equal(input_ids[0][content], labels[0][content])
    assert tokenizer.mask_token_id in input_ids[0].tolist()


def test_finetune_generator_dispatches_seq2seq():
    pytest.importorskip("transformers")

    tokenizer = _seq2seq_tokenizer()

    losses, last_epoch = finetune_generator(
        "molgen",
        _FakeLM(),
        tokenizer,
        ["[C][C][O]"],
        torch.device("cpu"),
        max_length=16,
        batch_size=1,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_norm_clip=1.0,
        verbose="none",
    )
    assert last_epoch == 0
    assert len(losses) == 1


def test_checkpoint_metadata_roundtrip(tmp_path):
    metadata = build_metadata(
        repo_id="chandar-lab/NovoMolGen_32M_SMILES_BPE",
        family="novomolgen",
        max_length=128,
        revision="hf-checkpoint",
        trust_remote_code=False,
        tokenizer_repo_id=None,
        generate_max_length=64,
        model_name="HFPretrainedMolecularGenerator",
    )
    save_metadata(str(tmp_path), metadata)
    loaded = load_metadata(str(tmp_path))
    assert loaded == metadata
    assert os.path.exists(os.path.join(tmp_path, METADATA_FILENAME))
    with open(os.path.join(tmp_path, METADATA_FILENAME), encoding="utf-8") as handle:
        on_disk = json.load(handle)
    assert on_disk["family"] == "novomolgen"
