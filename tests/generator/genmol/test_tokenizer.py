"""GenMol vocabulary, conversion, and input-integrity checks."""

import json
from importlib.util import find_spec

import pytest
import torch
from rdkit import Chem

if find_spec("safe") is None or find_spec("transformers") is None:
    pytest.skip("GenMol optional dependencies are not installed.", allow_module_level=True)

from torch_molecule.generator.genmol.tokenizer import (
    GenMolTokenizer,
    decode_safe,
    encode_scaffold,
    encode_smiles,
    filter_by_substructure,
)


@pytest.fixture(scope="module")
def tokenizer():
    return GenMolTokenizer()


def test_original_vocab_and_token_ids(tokenizer):
    assert len(tokenizer) == tokenizer.vocab_size == 1880
    assert (
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
    ) == (0, 1, 2, 3, 4)
    vocab = json.loads(tokenizer.to_json())["model"]["vocab"]
    assert tokenizer.encode("CCO") == [1, vocab["C"], vocab["C"], vocab["O"], 2]
    assert tokenizer.encode(".", add_special_tokens=False) == [11]


@pytest.mark.parametrize("value", ["CCO", "c1ccccc1", "[13CH3]O", "ClC1.CCO1"])
def test_tokenization_matches_safe_tokenizer(tokenizer, value):
    from safe.tokenizer import SAFETokenizer

    original = SAFETokenizer.from_dict(json.loads(tokenizer.to_json())).get_pretrained()
    assert tokenizer.encode(value) == original.encode(value)
    assert tokenizer.decode(tokenizer.encode(value)) == original.decode(
        original.encode(value), skip_special_tokens=True
    )


def test_padding_and_checkpoint_reconstruction(tokenizer, monkeypatch):
    import safe.tokenizer

    def unexpected_download(*args, **kwargs):
        pytest.fail("A bundled tokenizer must not download assets.")

    monkeypatch.setattr(safe.tokenizer, "hf_hub_download", unexpected_download)
    monkeypatch.setattr(safe.tokenizer, "urlopen", unexpected_download)
    restored = GenMolTokenizer(tokenizer.to_json())
    sequences = [restored.encode("C"), restored.encode("CCO")]
    batch = restored.pad(sequences)
    assert batch["input_ids"].dtype == torch.long
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch["input_ids"][0, -2:].tolist() == [3, 3]
    assert restored.batch_decode(batch["input_ids"]) == ["C", "CCO"]


@pytest.mark.parametrize("value", ["CCQ", "C C", "[rocket🚀]", ""])
def test_unsupported_input_is_not_silently_changed(tokenizer, value):
    with pytest.raises(ValueError):
        tokenizer.encode(value)


def test_overlong_input_is_not_truncated(tokenizer):
    with pytest.raises(ValueError, match="exceeding"):
        tokenizer.encode("CCO", max_len=4)
    assert tokenizer.encode("CCO", max_len=5)[-1] == tokenizer.eos_token_id


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "N[C@@H](C)C(=O)O"])
def test_conversion_matches_safe_library_and_preserves_molecule(smiles):
    import safe

    expected = safe.SAFEConverter(ignore_stereo=True).encoder(smiles, allow_empty=True)
    assert encode_smiles(smiles) == expected
    molecule = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(molecule)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(decode_safe(expected))) == Chem.MolToSmiles(molecule)


def test_fragment_encoding_preserves_native_attachment_handling():
    import safe

    fragment = "c1ccccc1[*]"
    assert encode_smiles(fragment, fragment=True) == safe.SAFEConverter(slicer=None).encoder(
        fragment, allow_empty=True
    )


@pytest.mark.parametrize("fragment", ["c1ccccc1", "CC(=O)N", "c1ccccc1[*]"])
def test_scaffold_preparation_matches_native_completion(fragment, monkeypatch):
    import random
    import safe

    monkeypatch.setattr(random, "choice", lambda options: sorted(options)[0])
    if "*" not in fragment:
        cores = safe.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
        expected_fragment = random.choice(cores)
    else:
        expected_fragment = fragment
    expected = (
        safe.SAFEConverter(ignore_stereo=True).encoder(expected_fragment, allow_empty=True) + "."
    )
    assert encode_scaffold(fragment) == expected


def test_scaffold_without_attachment_sites_rejected():
    with pytest.raises(ValueError, match="attachment sites"):
        encode_scaffold("[He]")


def test_invalid_smiles_rejected_and_invalid_generated_safe_dropped():
    for value in ("", "not-a-molecule", "C(C"):
        with pytest.raises(ValueError):
            encode_smiles(value)
    assert decode_safe("") is None
    assert decode_safe("not-a-molecule", fix=False) is None


def test_fragment_repair_and_substructure_filtering():
    import safe

    value = "CC.invalid"
    repaired = ".".join(
        part for part in value.split(".") if safe.decode(part, ignore_errors=True) is not None
    )
    assert decode_safe(value) == safe.decode(repaired, canonical=True, ignore_errors=True)
    assert filter_by_substructure(["CCO", "CCC", "c1ccccc1"], "[*]CO") == ["CCO"]
    with pytest.raises(ValueError, match="non-attachment"):
        filter_by_substructure(["CC"], "*")
