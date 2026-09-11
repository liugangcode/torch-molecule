import pytest

from torch_molecule.generator.pretrained.compat import ensure_safe_transformers_compat


def test_ensure_safe_transformers_compat_provides_constraints():
    pytest.importorskip("transformers")

    ensure_safe_transformers_compat()
    import transformers.generation as generation

    assert hasattr(generation, "DisjunctiveConstraint")
    assert hasattr(generation, "PhrasalConstraint")
