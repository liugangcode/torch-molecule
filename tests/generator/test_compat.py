from types import SimpleNamespace

import torch

from torch_molecule.generator.pretrained.compat import (
    patch_gp_molformer_generation_cache,
    to_legacy_past_key_values,
)


class _EmptyCache:
    def to_legacy_cache(self):
        return ((None, None), (None, None))


class _PopulatedCache:
    def to_legacy_cache(self):
        key = torch.zeros(1, 2, 3, 4)
        value = torch.zeros(1, 2, 3, 4)
        return ((key, value),)


def test_to_legacy_past_key_values_empty_cache_becomes_none():
    assert to_legacy_past_key_values(None) is None
    assert to_legacy_past_key_values(_EmptyCache()) is None
    assert to_legacy_past_key_values(((None, None),)) is None
    assert to_legacy_past_key_values(SimpleNamespace()) is None


def test_to_legacy_past_key_values_keeps_legacy_tensors():
    key = torch.zeros(1, 2, 3, 4)
    value = torch.zeros(1, 2, 3, 4)
    legacy = ((key, value),)
    assert to_legacy_past_key_values(legacy) == legacy
    converted = to_legacy_past_key_values(_PopulatedCache())
    assert converted[0][0].shape == (1, 2, 3, 4)


def test_patch_gp_molformer_generation_cache_converts_empty_cache():
    captured = {}

    class _Model:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)
            self.generation_config = SimpleNamespace(use_cache=True)

        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
            captured["past_key_values"] = past_key_values
            return {"input_ids": input_ids, "past_key_values": past_key_values}

    model = _Model()
    patch_gp_molformer_generation_cache(model)
    patch_gp_molformer_generation_cache(model)

    result = model.prepare_inputs_for_generation(
        torch.tensor([[1]]),
        past_key_values=_EmptyCache(),
    )
    assert captured["past_key_values"] is None
    assert result["past_key_values"] is None
    assert model.config.use_cache is False
    assert model.generation_config.use_cache is False
