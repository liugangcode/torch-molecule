"""Compatibility helpers for Hugging Face pretrained generators."""

from typing import Any


def ensure_safe_transformers_compat() -> None:
    """Restore generation constraint symbols removed in transformers 5.

    PyPI ``safe-mol`` still imports ``DisjunctiveConstraint`` and
    ``PhrasalConstraint`` from ``transformers.generation`` when the package
    is imported. Those classes were removed in transformers 5. Dummy
    stand-ins are enough to import ``safe.converter`` and run standard GPT-2
    ``generate()``. Official ``SAFEDesign`` constrained beam search is not used.
    """
    try:
        import transformers.generation as generation
    except ImportError:
        return

    if hasattr(generation, "DisjunctiveConstraint") and hasattr(generation, "PhrasalConstraint"):
        return

    class _RemovedConstraint:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "Constrained beam search was removed in transformers 5. "
                "SAFE-GPT in torch-molecule uses standard GPT-2 generate()."
            )

    if not hasattr(generation, "DisjunctiveConstraint"):
        generation.DisjunctiveConstraint = _RemovedConstraint
    if not hasattr(generation, "PhrasalConstraint"):
        generation.PhrasalConstraint = _RemovedConstraint
