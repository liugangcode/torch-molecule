import pytest

from torch_molecule.generator.pretrained.registry import resolve_family


@pytest.mark.parametrize(
    "repo_id,expected",
    [
        ("chandar-lab/NovoMolGen_32M_SMILES_BPE", "novomolgen"),
        ("ibm-research/GP-MoLFormer-Uniq", "gp_molformer"),
        ("zjunlp/MolGen-large", "molgen"),
        ("zjunlp/MolGen-large-opt", "molgen"),
        ("fairydance/molexar-10m-base", "molexar"),
        ("fairydance/molexar-10m-omni", "molexar"),
        ("some-user/custom-causal-lm", "causal_lm"),
    ],
)
def test_resolve_family(repo_id, expected):
    assert resolve_family(repo_id) == expected


def test_hf_pretrained_generator_requires_transformers():
    pytest.importorskip("transformers")

    from torch_molecule import HFPretrainedMolecularGenerator

    model = HFPretrainedMolecularGenerator(
        repo_id="chandar-lab/NovoMolGen_32M_SMILES_BPE",
    )
    assert model.repo_id == "chandar-lab/NovoMolGen_32M_SMILES_BPE"
    assert model.is_fitted_ is False


@pytest.mark.integration
def test_hf_pretrained_generator_novomolgen_smoke():
    pytest.importorskip("transformers")

    from torch_molecule import HFPretrainedMolecularGenerator

    model = HFPretrainedMolecularGenerator(
        repo_id="chandar-lab/NovoMolGen_32M_SMILES_BPE",
        generate_max_length=64,
    )
    model.fit()
    assert model.is_fitted_ is True

    smiles_list = model.generate(n_samples=2, temperature=1.0)
    assert isinstance(smiles_list, list)
    assert len(smiles_list) == 2
    assert all(isinstance(smiles, str) and smiles for smiles in smiles_list)


@pytest.mark.integration
def test_hf_pretrained_generator_finetune_smoke(tmp_path):
    pytest.importorskip("transformers")

    from torch_molecule import HFPretrainedMolecularGenerator

    model = HFPretrainedMolecularGenerator(
        repo_id="chandar-lab/NovoMolGen_32M_SMILES_BPE",
        batch_size=2,
        epochs=1,
        generate_max_length=32,
    )
    train_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    model.fit(train_smiles)

    assert model.is_fitted_ is True
    assert len(model.fitting_loss) == 1
    assert model.fitting_epoch == 0

    save_dir = tmp_path / "novomolgen-finetuned"
    model.save_to_local(str(save_dir))

    reloaded = HFPretrainedMolecularGenerator(
        repo_id="chandar-lab/NovoMolGen_32M_SMILES_BPE",
        generate_max_length=32,
    )
    reloaded.load_from_local(str(save_dir))
    assert reloaded.is_fitted_ is True
    assert reloaded.repo_id == model.repo_id
    assert reloaded._family == "novomolgen"

    smiles_list = reloaded.generate(n_samples=1, temperature=1.0, do_sample=True)
    assert isinstance(smiles_list, list)
    assert len(smiles_list) == 1
    assert isinstance(smiles_list[0], str)


@pytest.mark.integration
def test_hf_pretrained_generator_finetune_warns_on_y():
    pytest.importorskip("transformers")
    import numpy as np

    from torch_molecule import HFPretrainedMolecularGenerator

    model = HFPretrainedMolecularGenerator(
        repo_id="chandar-lab/NovoMolGen_32M_SMILES_BPE",
        batch_size=2,
        epochs=1,
    )
    with pytest.warns(UserWarning, match="Conditional fine-tuning"):
        model.fit(["CCO", "CC(=O)O"], y=np.array([0.0, 1.0]))
    assert model.is_fitted_ is True


def test_smiles_selfies_roundtrip():
    pytest.importorskip("selfies")

    from torch_molecule.generator.pretrained.utils import selfies_to_smiles, smiles_to_selfies

    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    recovered = selfies_to_smiles(smiles_to_selfies(smiles))
    assert recovered == smiles


def test_smiles_to_selfies_invalid_smiles():
    pytest.importorskip("selfies")

    from torch_molecule.generator.pretrained.utils import smiles_to_selfies

    with pytest.raises(ValueError, match="Invalid SMILES"):
        smiles_to_selfies(["not-a-smiles"])


def test_smiles_to_selfies_encoder_error_has_index(monkeypatch):
    pytest.importorskip("selfies")
    import selfies as sf

    from torch_molecule.generator.pretrained.utils import smiles_to_selfies

    def _raise_encoder(_smiles):
        raise sf.EncoderError("synthetic encoder failure")

    monkeypatch.setattr(sf, "encoder", _raise_encoder)
    with pytest.raises(ValueError, match="index 0.*not SELFIES-encodable"):
        smiles_to_selfies(["CCO"])


def test_selfies_to_smiles_drops_invalid_entries():
    pytest.importorskip("selfies")

    from torch_molecule.generator.pretrained.utils import selfies_to_smiles, smiles_to_selfies

    valid = smiles_to_selfies(["CCO"])[0]
    with pytest.warns(UserWarning, match="dropped 2 invalid SELFIES"):
        recovered = selfies_to_smiles([valid, "not-valid-selfies-[[[", ""])
    assert recovered == ["CCO"]
    assert "" not in recovered


def test_decode_outputs_molgen_drops_empty_and_warns():
    pytest.importorskip("transformers")
    pytest.importorskip("selfies")

    from torch_molecule import HFPretrainedMolecularGenerator
    from torch_molecule.generator.pretrained.utils import smiles_to_selfies

    model = HFPretrainedMolecularGenerator(repo_id="zjunlp/MolGen-large")
    model._family = "molgen"
    valid = smiles_to_selfies(["c1ccccc1"])[0]
    with pytest.warns(UserWarning, match="got 1/2 valid SMILES"):
        out = model._decode_outputs([valid, "not-a-selfies"])
    assert len(out) == 1
    assert "" not in out


def test_known_family_prefix_does_not_warn_unknown_repo():
    pytest.importorskip("transformers")
    import warnings

    from torch_molecule import HFPretrainedMolecularGenerator

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        HFPretrainedMolecularGenerator(repo_id="chandar-lab/NovoMolGen_157M")
    assert not any("Unknown repo_id" in str(item.message) for item in recorded)


def test_unknown_repo_fallback_warns():
    pytest.importorskip("transformers")

    from torch_molecule import HFPretrainedMolecularGenerator

    with pytest.warns(UserWarning, match="Unknown repo_id"):
        HFPretrainedMolecularGenerator(repo_id="some-user/custom-causal-lm")


def _transformers_supports_gp_molformer() -> bool:
    pytest.importorskip("transformers")
    import transformers

    major, minor, _ = map(int, transformers.__version__.split(".")[:3])
    return major < 5 and (major < 4 or minor < 57)


@pytest.mark.integration
def test_hf_pretrained_generator_gp_molformer_denovo():
    if not _transformers_supports_gp_molformer():
        pytest.skip("GP-MoLFormer requires transformers<=4.56.2")

    from torch_molecule import HFPretrainedMolecularGenerator

    model = HFPretrainedMolecularGenerator(
        repo_id="ibm-research/GP-MoLFormer-Uniq",
        generate_max_length=128,
    )
    model.fit()
    assert model.is_fitted_ is True

    smiles_list = model.generate(n_samples=2, temperature=1.0)
    assert isinstance(smiles_list, list)
    assert len(smiles_list) == 2
    assert all(isinstance(smiles, str) and smiles for smiles in smiles_list)


@pytest.mark.integration
def test_hf_pretrained_generator_gp_molformer_scaffold():
    if not _transformers_supports_gp_molformer():
        pytest.skip("GP-MoLFormer requires transformers<=4.56.2")

    from torch_molecule import HFPretrainedMolecularGenerator

    # IBM's official conditional prompt is a *partial* SMILES, not a closed ring.
    scaffold = "c1cccc"
    model = HFPretrainedMolecularGenerator(
        repo_id="ibm-research/GP-MoLFormer-Uniq",
        generate_max_length=128,
    )
    model.fit()

    smiles_list = model.generate(n_samples=2, scaffold=scaffold, temperature=1.0)
    assert isinstance(smiles_list, list)
    assert len(smiles_list) == 2
    assert all(isinstance(smiles, str) and smiles for smiles in smiles_list)
    assert all(smiles.startswith(scaffold) for smiles in smiles_list)


def test_gp_molformer_transformers_version_guard():
    pytest.importorskip("transformers")
    import transformers

    from torch_molecule import HFPretrainedMolecularGenerator

    major, minor, _ = map(int, transformers.__version__.split(".")[:3])
    model = HFPretrainedMolecularGenerator(
        repo_id="ibm-research/GP-MoLFormer-Uniq",
    )

    if major >= 5 or (major == 4 and minor >= 57):
        with pytest.raises(ImportError, match="transformers<=4.56.2"):
            model.fit()
    else:
        pytest.skip("GP-MoLFormer version guard only applies to transformers>=4.57")


def _molexar_available() -> bool:
    try:
        import molexar  # noqa: F401
        import fragment_selfies  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.integration
@pytest.mark.parametrize("repo_id", ["fairydance/molexar-10m-base"])
def test_hf_pretrained_generator_molexar_denovo(repo_id):
    if not _molexar_available():
        pytest.skip("Molexar requires fragment-selfies and molexar")

    from rdkit import Chem

    from torch_molecule import HFPretrainedMolecularGenerator

    model = HFPretrainedMolecularGenerator(repo_id=repo_id)
    model.fit()
    assert model.is_fitted_ is True

    smiles_list = model.generate(n_samples=2, temperature=0.8)
    assert isinstance(smiles_list, list)
    assert len(smiles_list) == 2
    assert all(isinstance(smiles, str) and smiles for smiles in smiles_list)
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in smiles_list)


@pytest.mark.integration
def test_hf_pretrained_generator_molexar_fragment_constraint():
    if not _molexar_available():
        pytest.skip("Molexar requires fragment-selfies and molexar")

    from rdkit import Chem

    from torch_molecule import HFPretrainedMolecularGenerator

    start_smiles = "[*]C1(CC#N)CN(S(=O)(=O)CC)C1"
    model = HFPretrainedMolecularGenerator(repo_id="fairydance/molexar-10m-base")
    model.fit()

    smiles_list = model.generate(
        n_samples=2,
        start_smiles=start_smiles,
        generation_task="motif_extension",
        temperature=0.8,
    )
    assert len(smiles_list) == 2
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in smiles_list)


def test_smiles_fragment_selfies_roundtrip():
    if not _molexar_available():
        pytest.skip("Molexar requires fragment-selfies and molexar")

    from torch_molecule.generator.pretrained.utils import (
        fragment_selfies_to_smiles,
        smiles_to_fragment_selfies,
    )

    smiles = ["CCO", "c1ccccc1"]
    recovered = fragment_selfies_to_smiles(smiles_to_fragment_selfies(smiles))
    assert len(recovered) == 2
    assert all(smiles_string for smiles_string in recovered)


@pytest.mark.parametrize(
    "repo_id",
    [
        "zjunlp/MolGen-large",
        "zjunlp/MolGen-large-opt",
    ],
)
@pytest.mark.integration
def test_hf_pretrained_generator_molgen_smoke(repo_id):
    pytest.importorskip("transformers")
    pytest.importorskip("selfies")
    from rdkit import Chem

    from torch_molecule import HFPretrainedMolecularGenerator

    model = HFPretrainedMolecularGenerator(
        repo_id=repo_id,
        generate_max_length=20,
    )
    model.fit()
    assert model.is_fitted_ is True

    smiles_list = model.generate(n_samples=2, num_beams=5)
    assert isinstance(smiles_list, list)
    assert len(smiles_list) == 2
    assert all(isinstance(smiles, str) and smiles for smiles in smiles_list)
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in smiles_list)


@pytest.mark.integration
def test_hf_pretrained_generator_molgen_scaffold_prefix():
    pytest.importorskip("transformers")
    pytest.importorskip("selfies")
    from rdkit import Chem

    from torch_molecule import HFPretrainedMolecularGenerator
    from torch_molecule.generator.pretrained.utils import smiles_to_selfies

    scaffold = "c1ccccc1"
    benzene = Chem.MolFromSmiles(scaffold)
    prefix_selfies = smiles_to_selfies([scaffold])[0]

    model = HFPretrainedMolecularGenerator(
        repo_id="zjunlp/MolGen-large",
        generate_max_length=20,
    )
    model.fit()

    smiles_from_scaffold = model.generate(n_samples=2, scaffold=scaffold, num_beams=5)
    smiles_from_prefix = model.generate(n_samples=2, prefix_selfies=prefix_selfies, num_beams=5)

    assert len(smiles_from_scaffold) == 2
    assert len(smiles_from_prefix) == 2
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in smiles_from_scaffold)
    assert all(
        Chem.MolFromSmiles(smiles) is not None
        and Chem.MolFromSmiles(smiles).HasSubstructMatch(benzene)
        for smiles in smiles_from_scaffold
    )
