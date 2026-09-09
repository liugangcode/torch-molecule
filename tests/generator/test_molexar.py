import importlib.util

import pytest

from torch_molecule.generator.pretrained.families.molexar import (
    extract_conditions,
    resolve_start_string,
)


def test_resolve_start_string_none_for_de_novo():
    assert resolve_start_string() is None


def test_resolve_start_string_accepts_literal_prefix():
    prefix = "[Frag][C][C][Attach:0]"
    assert resolve_start_string(start_string=prefix) == prefix


@pytest.mark.skipif(
    importlib.util.find_spec("molexar") is None,
    reason="molexar not installed",
)
def test_resolve_start_string_from_smiles_fragment():
    start_smiles = "[*]C1(CC#N)CN(S(=O)(=O)CC)C1"
    resolved = resolve_start_string(
        start_smiles=start_smiles,
        generation_task="motif_extension",
    )
    assert resolved is not None
    assert "[Attach:0]" in resolved


def test_extract_conditions_from_kwargs():
    kwargs = {"temperature": 0.8, "mol_qed": 0.9, "conditions": {"mol_logp": 2.5}}
    conditions = extract_conditions(kwargs)
    assert conditions == {"mol_logp": 2.5, "mol_qed": 0.9}
    assert kwargs == {"temperature": 0.8}
