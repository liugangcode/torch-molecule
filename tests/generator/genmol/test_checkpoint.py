"""Saving, loading, and native GenMol checkpoint import."""

from types import SimpleNamespace

import pytest
import torch

from torch_molecule import GenMolMolecularGenerator

from .utils import SMILES, seed, small_model


@pytest.mark.parametrize("ema_decay", [0.0, 0.9])
def test_checkpoint_preserves_weights_ema_and_generation_metadata(
    tmp_path, ema_decay, genmol_dependencies
):
    seed()
    model = small_model(ema_decay=ema_decay)
    model.fit(SMILES)
    path = str(tmp_path / "genmol.pt")
    model.save_to_local(path)
    restored = GenMolMolecularGenerator(device="cpu")
    restored.load_from_local(path)
    assert restored.is_fitted_
    assert restored.hidden_size == model.hidden_size
    assert restored.sequence_lengths_ == model.sequence_lengths_
    assert restored.tokenizer_json_ == model.tokenizer_json_
    for original, actual in zip(model.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    if ema_decay:
        assert restored.model.ema.num_updates == model.model.ema.num_updates
        for original, actual in zip(
            model.model.ema.shadow_params, restored.model.ema.shadow_params
        ):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
    else:
        assert restored.model.ema is None


def test_native_checkpoint_restores_backbone_and_ema(tmp_path, genmol_dependencies):
    seed()
    source = small_model()
    source.fit(SMILES)
    path = tmp_path / "native.ckpt"
    torch.save(
        {
            "state_dict": {
                "backbone." + key: value
                for key, value in source.model.backbone.state_dict().items()
            },
            "ema": source.model.ema.state_dict(),
            "hyper_parameters": {"config": {"training": {"use_bracket_safe": False}}},
        },
        path,
    )
    restored = small_model()
    assert (
        restored.load_from_pretrained(path, sequence_lengths=source.sequence_lengths_) is restored
    )
    assert restored.is_fitted_
    assert restored.sequence_lengths_ == source.sequence_lengths_
    for original, actual in zip(
        source.model.backbone.parameters(), restored.model.backbone.parameters()
    ):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    for original, actual in zip(source.model.ema.shadow_params, restored.model.ema.shadow_params):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_native_raw_checkpoint_and_invalid_load_preserve_state(tmp_path, genmol_dependencies):
    seed()
    source = small_model(ema_decay=0)
    source.fit(SMILES)
    path = tmp_path / "native_raw.ckpt"
    state = {"backbone." + key: value for key, value in source.model.backbone.state_dict().items()}
    torch.save({"state_dict": state}, path)
    restored = small_model(ema_decay=0)
    restored.load_from_pretrained(path, sequence_lengths=source.sequence_lengths_)
    assert restored.model.ema is None
    original = restored.model
    torch.save({"state_dict": {}}, path)
    with pytest.raises(RuntimeError):
        restored.load_from_pretrained(path, sequence_lengths=source.sequence_lengths_)
    assert restored.model is original and restored.is_fitted_
    torch.save(
        {
            "state_dict": state,
            "hyper_parameters": {"config": {"training": {"use_bracket_safe": True}}},
        },
        path,
    )
    with pytest.raises(ValueError, match="V2"):
        restored.load_from_pretrained(path, sequence_lengths=source.sequence_lengths_)


def test_native_checkpoint_rejects_incompatible_attention_heads(tmp_path, genmol_dependencies):
    source = small_model(epochs=1)
    source.fit(SMILES)
    checkpoint = {
        "state_dict": {
            "backbone." + key: value for key, value in source.model.backbone.state_dict().items()
        },
        "ema": source.model.ema.state_dict(),
        "hyper_parameters": {"config": {"model": {"num_attention_heads": 4}}},
    }
    path = tmp_path / "wrong_heads.ckpt"
    torch.save(checkpoint, path)
    restored = small_model()
    with pytest.raises(ValueError, match="num_attention_heads"):
        restored.load_from_pretrained(path, sequence_lengths=source.sequence_lengths_)
    assert not restored.is_fitted_ and restored.model is None


def test_checkpoint_rejects_pickled_configuration_objects(tmp_path, genmol_dependencies):
    source = small_model(epochs=1)
    source.fit(SMILES)
    original = source.model
    path = tmp_path / "object_config.ckpt"
    torch.save(
        {
            "state_dict": {
                "backbone." + key: value
                for key, value in source.model.backbone.state_dict().items()
            },
            "ema": source.model.ema.state_dict(),
            "hyper_parameters": {
                "config": SimpleNamespace(model=source.model.backbone.config.to_dict())
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="weights-only export"):
        source.load_from_pretrained(path, sequence_lengths=source.sequence_lengths_)
    assert source.model is original and source.is_fitted_
