"""Offline integration tests using locally initialized Transformers models."""

import warnings

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from torch_molecule import HFPretrainedMolecularGenerator
from torch_molecule.generator.pretrained.compat import ensure_safe_transformers_compat
from torch_molecule.generator.pretrained.finetune import corrupt_token_ids


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    """Build tiny causal and seq2seq checkpoints without network access."""
    def create(family):
        path = tmp_path / family
        vocabulary = {
            token: index
            for index, token in enumerate([
                "<pad>", "<s>", "</s>", "<unk>", "<mask>",
                "C", "CC", "CCO", "CCN", "[C]", "[C][C]", "[C][C][O]",
            ])
        }
        backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
        backend.post_processor = tokenizers.processors.TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)],
        )
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", bos_token="<s>",
            eos_token="</s>", unk_token="<unk>", mask_token="<mask>",
            model_input_names=["input_ids", "attention_mask"],
        )
        safe_tokenizer = None
        if family == "safe_gpt":
            ensure_safe_transformers_compat()
            safe_tokenizer_module = pytest.importorskip("safe.tokenizer")
            safe_tokenizer = safe_tokenizer_module.SAFETokenizer()
            training_file = tmp_path / "safe.txt"
            training_file.write_text("CC\nCCO\nCCN\n")
            safe_tokenizer.train([str(training_file)])
            tokenizer = safe_tokenizer.get_pretrained()
        torch.manual_seed(7)
        if family == "molgen":
            model = transformers.BartForConditionalGeneration(transformers.BartConfig(
                vocab_size=len(vocabulary), d_model=16, encoder_layers=1, decoder_layers=1,
                encoder_attention_heads=2, decoder_attention_heads=2,
                encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32,
                pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
            ))
            repo_id = "zjunlp/MolGen-large"
        else:
            model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
                vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2, n_positions=32,
                pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ))
            repo_id = ("datamol-io/safe-gpt" if family == "safe_gpt"
                       else "chandar-lab/NovoMolGen_32M_SMILES_BPE")
        model.save_pretrained(path)
        if safe_tokenizer is not None:
            safe_tokenizer.save_pretrained(path)
        else:
            tokenizer.save_pretrained(path)
        generator = HFPretrainedMolecularGenerator(
            repo_id=repo_id, max_length=16, generate_max_length=8,
            batch_size=2, epochs=1, device="cpu",
        )
        generator._family = family
        generator._load_pretrained(local_path=str(path))
        generator.is_fitted_ = True
        load_pretrained = generator._load_pretrained
        monkeypatch.setattr(generator, "_load_pretrained", lambda: load_pretrained(local_path=str(path)))
        return generator

    return create


@pytest.mark.parametrize("family", ["novomolgen", "molgen", "safe_gpt"])
def test_finetune_and_checkpoint_roundtrip(local_model, tmp_path, family):
    if family == "molgen":
        pytest.importorskip("selfies")
    generator = local_model(family)
    before = {key: value.detach().clone() for key, value in generator.model.named_parameters()}
    generator.fit(["CC", "CCO"])
    assert len(generator.fitting_loss) == 1
    assert np.isfinite(generator.fitting_loss).all()
    assert any(not torch.equal(before[key], value) for key, value in generator.model.named_parameters())
    assert not generator.model.training

    options = {"do_sample": False}
    if family == "molgen":
        options.update(prefix_selfies="[C]", min_length=2, num_beams=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = generator.generate(n_samples=1, **options)
    checkpoint = tmp_path / "saved"
    generator.save_to_local(str(checkpoint))
    restored = HFPretrainedMolecularGenerator(repo_id=generator.repo_id, device="cpu")
    restored.load_from_local(str(checkpoint))
    assert restored.get_params()["generate_max_length"] == 8
    assert restored._family == family
    for key, value in generator.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], value, rtol=0, atol=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert restored.generate(n_samples=1, **options) == expected


def test_molgen_corruption_preserves_special_tokens(local_model):
    generator = local_model("molgen")
    input_ids = torch.tensor([[1, 9, 10, 2, 0]])
    result = corrupt_token_ids(input_ids, generator.tokenizer, mask_prob=1.0)
    assert result.tolist() == [[1, 4, 4, 2, 0]]
    assert input_ids.tolist() == [[1, 9, 10, 2, 0]]


def test_novomolgen_prefix_uses_training_start_token(local_model, monkeypatch):
    generator = local_model("novomolgen")
    received = {}
    generate = generator.model.generate

    def capture(**kwargs):
        received.update(kwargs)
        return generate(**kwargs)

    monkeypatch.setattr(generator.model, "generate", capture)
    result = generator.generate(n_samples=1, scaffold="CC", max_new_tokens=3, do_sample=False)
    assert received["input_ids"].tolist() == [[generator.tokenizer.bos_token_id, 6]]
    assert received["max_new_tokens"] == 3
    assert result[0].startswith("CC")
