"""Compatibility helpers for Hugging Face pretrained generators."""

import sys
import types
from typing import Any, Optional, Tuple


def _parse_transformers_version(version: str) -> Tuple[int, int, int]:
    parts = version.split(".")
    return tuple(int(part) for part in parts[:3])


def ensure_gp_molformer_transformers_compat() -> None:
    """Validate that the installed ``transformers`` version can load GP-MoLFormer."""
    import transformers

    major, minor, _ = _parse_transformers_version(transformers.__version__)
    if major >= 5 or (major == 4 and minor >= 57):
        raise ImportError(
            "GP-MoLFormer remote code is not compatible with transformers "
            f"{transformers.__version__}. Install transformers<=4.56.2, for example:\n"
            "  pip install 'transformers>=4.40,<=4.56.2'"
        )


def ensure_transformers_onnx_compat() -> None:
    """Provide a stub ``transformers.onnx`` module for legacy remote code.

    IBM MoLFormer remote configs import ``OnnxConfig`` from ``transformers.onnx``,
    which was removed in recent ``transformers`` releases. The ONNX export class
    is not needed for generation, so a lightweight stub is sufficient.
    """
    if "transformers.onnx" in sys.modules:
        return

    try:
        from transformers.onnx import OnnxConfig  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    onnx_module = types.ModuleType("transformers.onnx")

    class OnnxConfig:
        """Minimal stub for legacy remote configuration modules."""

    onnx_module.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = onnx_module


def to_legacy_past_key_values(past_key_values: Any) -> Optional[tuple]:
    """Convert HF Cache objects to the tuple layout IBM MoLFormer expects.

    ``transformers`` 4.47+ injects an empty ``DynamicCache`` into
    ``prepare_inputs_for_generation``. IBM remote code then does
    ``past_key_values[0][0].shape``, which raises because the empty cache
    stores ``None`` instead of tensors.
    """
    if past_key_values is None:
        return None

    if not isinstance(past_key_values, (tuple, list)):
        converter = getattr(past_key_values, "to_legacy_cache", None)
        if converter is None:
            return None
        past_key_values = converter()

    if not past_key_values:
        return None
    first_layer = past_key_values[0]
    if not first_layer:
        return None
    if first_layer[0] is None:
        return None
    return tuple(past_key_values)


def patch_gp_molformer_generation_cache(model: Any) -> None:
    """Make GP-MoLFormer generation tolerate DynamicCache from transformers 4.56.

    IBM ``MolformerForCausalLM.prepare_inputs_for_generation`` only understands
    the legacy tuple KV cache. Convert Cache objects (or empty caches) before
    that method runs, and default ``use_cache`` off so later steps do not
    reintroduce an incompatible cache layout.
    """
    if getattr(model, "_gp_molformer_cache_patched", False):
        return

    original = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation(
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        past_key_values = to_legacy_past_key_values(past_key_values)
        return original(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    model.prepare_inputs_for_generation = prepare_inputs_for_generation
    model._gp_molformer_cache_patched = True

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = False
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False
