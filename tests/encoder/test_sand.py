"""Self-contained SAND training, persistence, and optional upstream parity."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from torch_molecule import SANDMolecularEncoder
from torch_molecule.base import BaseMolecularEncoder

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "CCCCCCCC",
    "c1ccncc1",
    "C1CCCCC1",
    "CC(=O)Nc1ccccc1",
]
PAIRS = np.array([[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 7]])
SCORES = np.linspace(0.1, 0.9, len(PAIRS), dtype=np.float32)
SEED = 7


@pytest.fixture
def small_params():
    def make(compressed=False):
        return {
            "num_layer": 2,
            "hidden_size": 64,
            "emb_dim": 64,
            "quantize_emb": compressed,
            "num_pq_quantizers": 4,
            "ivf_codebook_size": 64,
            "warmup_iters": 0,
        }

    return make


def _local(params):
    estimator = SANDMolecularEncoder(**params, device="cpu")
    return estimator.model_class(**estimator._get_model_params())


def _native(params):
    sand = pytest.importorskip("sand")
    from sand.nn.backbone import IVFPQEncoderBackbone
    from sand.output.embedding import QuantizationOutput, SpearmanROutput, UniformityOutput

    settings = SANDMolecularEncoder(**params, device="cpu").get_params()
    backbone = IVFPQEncoderBackbone(
        emb_dim=settings["emb_dim"],
        encoder_hidden_dim=settings["hidden_size"],
        encoder_num_layers=settings["num_layer"],
        encoder_type=settings["encoder_type"],
        num_atom_features=settings["num_atom_features"],
        num_edge_features=settings["num_edge_features"],
        normalize_shared_emb=settings["normalize_shared_emb"],
        quantize_emb=settings["quantize_emb"],
        num_ivf_quantizers=settings["num_ivf_quantizers"],
        num_pq_quantizers=settings["num_pq_quantizers"],
        ivf_codebook_size=settings["ivf_codebook_size"],
        ivf_threshold_ema_dead_code=settings["ivf_threshold_ema_dead_code"],
        ivf_ema_decay=settings["ivf_ema_decay"],
    )
    outputs = [
        SpearmanROutput(
            sim_key="score",
            emb_key="emb_quantized" if settings["quantize_emb"] else "emb",
            loss_weight=settings["similarity_loss_weight"],
            regularization_strength=settings["regularization_strength"],
        ),
        UniformityOutput(
            name="uniformity" if settings["quantize_emb"] else "uniformity2",
            emb_key="emb",
            loss_weight=settings["uniformity_loss_weight"],
            num_matches=settings["uniformity_num_matches"],
            margin=settings["uniformity_margin"],
        ),
    ]
    if settings["quantize_emb"]:
        outputs.extend(
            [
                QuantizationOutput(
                    loss_weight=settings["ivf_loss_weight"],
                    nq_key="emb",
                    q_key="ivf_x",
                    name="quantization_ivf",
                    use_cosine=True,
                    add_dataloader_idx_to_name=True,
                ),
                QuantizationOutput(
                    loss_weight=settings["pq_loss_weight"],
                    nq_key="x_res",
                    q_key="pq_x_res",
                    name="quantization_pq",
                    use_cosine=False,
                    add_dataloader_idx_to_name=True,
                ),
            ]
        )
    return sand.SANDModel(
        backbone=backbone,
        outputs=outputs,
        lr=settings["learning_rate"],
        min_lr=settings["min_lr"],
        weight_decay=settings["weight_decay"],
        warmup_iters=settings["warmup_iters"],
        decay_iters=settings["decay_iters"],
    )


def test_training_and_persistence_without_upstream_or_lightning(tmp_path):
    script = """
import importlib.abc
import sys
blocked = {'sand', 'lightning', 'pytorch_lightning', 'lightning_fabric',
           'faiss', 'jsonargparse', 'zarr', 'yaml'}
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Deliberately unavailable: ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockImports())
import numpy as np
import torch
from torch_molecule import SANDMolecularEncoder
from torch_molecule.encoder.sand.model import SANDModel
assert issubclass(SANDModel, torch.nn.Module)
assert all(base.__module__.split('.')[0] not in blocked for base in SANDModel.__mro__)
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCCCCCC', 'c1ccncc1',
          'C1CCCCC1', 'CC(=O)Nc1ccccc1']
pairs = [[0,1], [0,2], [0,3], [1,4], [2,5], [3,6], [4,7], [5,7]]
torch.manual_seed(7)
model = SANDMolecularEncoder(num_layer=2, hidden_size=64, emb_dim=64,
    quantize_emb=True, num_pq_quantizers=4, ivf_codebook_size=64, warmup_iters=0,
    batch_size=8, epochs=2, device='cpu')
model.fit(smiles, np.linspace(.1, .9, 8, dtype=np.float32), pairs=pairs)
assert len(model.fitting_loss) == 2 and np.isfinite(model.fitting_loss).all()
expected = model.encode(smiles)
assert expected.shape == (8, 64) and torch.isfinite(expected).all()
model.save_to_local(sys.argv[1])
restored = SANDMolecularEncoder(device='cpu')
restored.load_from_local(sys.argv[1])
torch.testing.assert_close(restored.encode(smiles), expected)
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "standalone.pt")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_estimator_parameters_and_unfitted_state():
    model = SANDMolecularEncoder(batch_size=2, epochs=2, device="cpu")
    assert isinstance(model, BaseMolecularEncoder)
    assert "checkpoint_path" not in model.get_params()
    assert "config" not in model.get_params()
    assert (model.num_layer, model.hidden_size, model.emb_dim) == (7, 1024, 512)
    assert model.quantize_emb and model.num_pq_quantizers == 32
    assert model.get_params()["epochs"] == 2
    assert model.fitting_loss == []
    assert model.set_params(batch_size=3, learning_rate=0.001) is model
    assert model.batch_size == 3 and model.learning_rate == 0.001
    with pytest.raises(ValueError):
        model.set_params(unknown_parameter=True)
    with pytest.raises(AttributeError, match="fit"):
        model.encode(SMILES)
    with pytest.raises(ValueError, match="fitted"):
        model.save_to_local("unfitted.pt")
    with pytest.raises(ValueError):
        model.fit(SMILES)


def test_explicit_parameters_reach_backbone_losses_and_schedule(small_params):
    params = small_params(compressed=True)
    params.update(
        num_layer=3,
        hidden_size=48,
        emb_dim=32,
        num_atom_features=64,
        num_edge_features=18,
        normalize_shared_emb=False,
        similarity_loss_weight=0.8,
        regularization_strength=0.7,
        uniformity_loss_weight=0.4,
        uniformity_margin=0.3,
        ivf_loss_weight=0.2,
        pq_loss_weight=0.1,
        learning_rate=0.002,
        min_lr=0.00002,
        warmup_iters=3,
        decay_iters=10,
    )
    estimator = SANDMolecularEncoder(**small_params(compressed=True), device="cpu")
    assert estimator.set_params(**params) is estimator
    assert all(estimator.get_params()[name] == value for name, value in params.items())
    model = estimator.model_class(**estimator._get_model_params())
    encoder = model.backbone.encoder.model
    assert encoder.hidden_dim == 48 and encoder.out_dim == 32
    assert encoder.gnn.num_layers == 3
    assert encoder.atom_map.embedding.num_embeddings == 65
    assert encoder.edge_map.embedding.num_embeddings == 19
    assert not model.backbone.normalize_shared_emb
    assert [output.loss_weight for output in model.outputs] == [0.8, 0.4, 0.2, 0.1]
    assert model.outputs[0].regularization_strength == 0.7
    assert model.outputs[1].margin == 0.3
    assert model.outputs[1].num_matches == estimator.uniformity_num_matches
    assert (model.lr, model.min_lr, model.warmup_iters, model.decay_iters) == (
        0.002,
        0.00002,
        3,
        10,
    )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_constructor_batch_size(batch_size):
    with pytest.raises(ValueError):
        SANDMolecularEncoder(batch_size=batch_size)


@pytest.mark.parametrize("compressed", [False, True], ids=["dense", "compressed"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scratch_training_updates_weights_and_native_ema(small_params, compressed, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    params = small_params(compressed)
    torch.manual_seed(SEED)
    initial = _local(params).state_dict()
    model = SANDMolecularEncoder(
        **params, batch_size=len(PAIRS), epochs=2, device=device, grad_clip_value=1.0
    )
    torch.manual_seed(SEED)
    with patch("torch.load", side_effect=AssertionError("Scratch training loaded weights")):
        assert model.fit(SMILES, SCORES, pairs=PAIRS) is model
    assert model.is_fitted_
    assert next(model.model.parameters()).device.type == device
    assert len(model.fitting_loss) == 2 and np.isfinite(model.fitting_loss).all()
    assert model.fitting_epoch == 1
    trained = model.model.state_dict()
    assert any(
        not torch.equal(initial[key], value.detach().cpu())
        for key, value in trained.items()
        if key.startswith("backbone.encoder")
    )
    gradients = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) for grad in gradients)
    if compressed:
        assert any(
            not torch.equal(initial[key], value.detach().cpu())
            for key, value in trained.items()
            if key.endswith("centroids")
        )
    encoded = model.encode(SMILES)
    expected = model.model.encode_smiles(SMILES, batch_size=len(PAIRS)).detach().cpu()
    assert encoded.shape == (len(SMILES), 64) and torch.isfinite(encoded).all()
    assert encoded.device.type == "cpu" and not encoded.requires_grad
    torch.testing.assert_close(encoded, expected)
    np.testing.assert_allclose(model.encode(SMILES, "np"), encoded.numpy(), atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("compressed", [False, True], ids=["dense", "compressed"])
def test_native_first_step_loss_gradients_update_and_ema_match(small_params, compressed):
    pytest.importorskip("sand")
    from torch_molecule.encoder.sand.model import SANDModel

    params = small_params(compressed)
    batches = []
    original_loss = SANDModel.compute_loss

    def capture_input(module, batch):
        batches.append(copy.deepcopy(batch))
        return original_loss(module, batch)

    model = SANDMolecularEncoder(**params, batch_size=len(PAIRS), epochs=1, device="cpu")
    torch.manual_seed(SEED)
    with patch.object(SANDModel, "compute_loss", capture_input):
        model.fit(SMILES, SCORES, pairs=PAIRS)
    assert len(batches) == 1
    assert batches[0]["batch_size"] == len(PAIRS)
    assert batches[0]["num_graphs"] == 2 * len(PAIRS)
    torch.testing.assert_close(batches[0]["score"].sort().values, torch.from_numpy(SCORES))
    torch.manual_seed(SEED)
    native = _native(params).train()
    optimizer = native.configure_optimizers()
    batch = native(batches[0])
    loss = sum(
        output(batch, stage="train", log_func=lambda **kwargs: None, dataloader_idx=0)
        for output in native.outputs
    )
    if compressed:
        native.backbone.ema_update(batch)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert model.fitting_loss[0] == pytest.approx(loss.item(), abs=2e-5)
    for name, expected in native.named_parameters():
        actual = dict(model.model.named_parameters())[name]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
    for name, expected in native.state_dict().items():
        torch.testing.assert_close(model.model.state_dict()[name], expected, atol=2e-6, rtol=2e-5)


@pytest.mark.integration
def test_native_multiepoch_warmup_and_cosine_schedule_match(small_params):
    pytest.importorskip("sand")
    from lightning.pytorch import Callback, Trainer
    from torch.utils.data import DataLoader, IterableDataset

    from torch_molecule.encoder.sand.model import SANDModel

    params = small_params()
    params.update(warmup_iters=2, decay_iters=4)
    batches, local_lrs = [], []
    original_loss = SANDModel.compute_loss
    original_step = torch.optim.AdamW.step

    def capture_batch(module, batch):
        batches.append(copy.deepcopy(batch))
        return original_loss(module, batch)

    def capture_lr(optimizer, *args, **kwargs):
        local_lrs.append(optimizer.param_groups[0]["lr"])
        return original_step(optimizer, *args, **kwargs)

    model = SANDMolecularEncoder(**params, batch_size=3, epochs=2, device="cpu")
    torch.manual_seed(SEED)
    with patch.object(SANDModel, "compute_loss", capture_batch), patch.object(
        torch.optim.AdamW, "step", capture_lr
    ):
        model.fit(SMILES, SCORES, pairs=PAIRS)
    assert len(batches) == len(local_lrs) == 6
    steps_per_epoch = len(batches) // model.epochs

    class ReplayBatches(IterableDataset):
        epoch = 0

        def __len__(self):
            return steps_per_epoch

        def __iter__(self):
            start = self.epoch * steps_per_epoch
            self.epoch += 1
            yield from copy.deepcopy(batches[start : start + steps_per_epoch])

    class RecordNative(Callback):
        def __init__(self):
            self.lrs = []
            self.losses = []

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            self.lrs.append(optimizer.param_groups[0]["lr"])

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.losses.append(float(outputs["loss"].detach()))

    torch.manual_seed(SEED)
    native = _native(params)
    recorded = RecordNative()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[recorded],
        log_every_n_steps=1,
    )
    trainer.fit(native, train_dataloaders=DataLoader(ReplayBatches(), batch_size=None))
    np.testing.assert_allclose(local_lrs, recorded.lrs, rtol=0, atol=1e-12)
    assert local_lrs[0] == model.learning_rate
    assert min(local_lrs[1:]) < model.learning_rate
    assert model.fitting_loss == pytest.approx(
        [np.mean(recorded.losses[:3]), np.mean(recorded.losses[3:])], abs=2e-5
    )
    for name, expected in native.state_dict().items():
        torch.testing.assert_close(model.model.state_dict()[name], expected, atol=2e-6, rtol=2e-5)


def test_refit_initializes_fresh_model(small_params):
    params = small_params()
    model = SANDMolecularEncoder(**params, batch_size=len(PAIRS), epochs=1, device="cpu")
    model.set_params(device="cpu")
    torch.manual_seed(SEED)
    model.fit(SMILES, SCORES, pairs=PAIRS)
    first_model = model.model
    first_state = copy.deepcopy(first_model.state_dict())
    first_history = model.fitting_loss.copy()
    torch.manual_seed(SEED)
    model.fit(SMILES, SCORES, pairs=PAIRS)
    assert model.model is not first_model
    assert model.fitting_loss == pytest.approx(first_history)
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, first_state[key])


def test_trained_checkpoint_preserves_flat_attributes_and_encoding_order(small_params, tmp_path):
    params = small_params()
    params.update(
        similarity_loss_weight=0.7,
        uniformity_loss_weight=0.3,
        regularization_strength=0.8,
        min_lr=2e-5,
        warmup_iters=1,
        decay_iters=7,
    )
    model = SANDMolecularEncoder(**params, batch_size=3, epochs=2, device="cpu")
    model.fit(SMILES, SCORES, pairs=PAIRS)
    expected = model.encode(SMILES)
    saved_path = tmp_path / "sand.pt"
    model.save_to_local(str(saved_path))
    checkpoint = torch.load(saved_path, weights_only=False)
    assert "config" not in checkpoint["hyperparameters"]
    assert not any(isinstance(value, dict) for value in checkpoint["hyperparameters"].values())
    restored = SANDMolecularEncoder(device="cpu")
    restored.load_from_local(str(saved_path))
    for name, value in model.get_params().items():
        if name not in {"device", "model_name"}:
            assert restored.get_params()[name] == value, name
    restored.set_params(batch_size=1)
    torch.testing.assert_close(
        restored.encode(SMILES[::-1]), expected.flip(0), atol=1e-6, rtol=1e-5
    )


def test_training_handles_singleton_tail_and_tied_score_batches(small_params):
    from torch_molecule.encoder.sand.model import SANDModel

    params = small_params()
    observed = []
    original_step = SANDModel.compute_loss

    def record_scores(module, batch, *args, **kwargs):
        observed.append(batch["score"].clone())
        return original_step(module, batch, *args, **kwargs)

    model = SANDMolecularEncoder(**params, batch_size=2, epochs=1, device="cpu")
    torch.manual_seed(SEED)
    with patch.object(SANDModel, "compute_loss", record_scores):
        model.fit(SMILES, np.array([0.1, 0.1, 0.1, 0.1, 0.9]), pairs=PAIRS[:5])
    assert len(observed) == 2
    assert sum(len(scores) for scores in observed) >= 5
    assert all(len(scores) >= 2 and scores.unique().numel() >= 2 for scores in observed)
    assert np.isfinite(model.fitting_loss).all()


def test_training_data_and_parameter_validation(small_params):
    params = small_params()
    invalid = [
        (SMILES, None, PAIRS),
        (SMILES, SCORES, None),
        ([], SCORES, PAIRS),
        (["bad-smiles"], SCORES, PAIRS),
        (SMILES, SCORES[:-1], PAIRS),
        (SMILES, SCORES[:, None], PAIRS),
        (SMILES, np.ones(len(PAIRS)), PAIRS),
        (SMILES, np.full(len(PAIRS), np.nan), PAIRS),
        (SMILES, np.full(len(PAIRS), np.inf), PAIRS),
        (SMILES, SCORES, PAIRS.ravel()),
        (SMILES, SCORES, PAIRS.astype(float)),
        (SMILES, SCORES, PAIRS - 10),
        (SMILES, SCORES, PAIRS + len(SMILES)),
        (SMILES, SCORES[:1], PAIRS[:1]),
    ]
    for molecules, scores, pairs in invalid:
        model = SANDMolecularEncoder(**params, epochs=1, device="cpu")
        with pytest.raises(ValueError):
            model.fit(molecules, scores, pairs=pairs)
        assert not model.is_fitted_
    for invalid_params in ({"batch_size": 1}, {"epochs": 0}, {"learning_rate": 0}):
        model = SANDMolecularEncoder(**params, epochs=1, device="cpu")
        model.set_params(**invalid_params)
        with pytest.raises(ValueError):
            model.fit(SMILES, SCORES, pairs=PAIRS)


def test_encoding_validation_after_training(small_params):
    params = small_params()
    model = SANDMolecularEncoder(**params, epochs=1, device="cpu")
    model.fit(SMILES, SCORES, pairs=PAIRS)
    for invalid in ([], ["invalid-smiles"], [None], "CCO"):
        with pytest.raises(ValueError):
            model.encode(invalid)
    with pytest.raises(ValueError):
        model.encode(SMILES, return_type="invalid")
    model.set_params(batch_size=0)
    with pytest.raises(ValueError):
        model.encode(SMILES)


def test_pretrained_checkpoint_loading_is_explicit(small_params, tmp_path):
    params = small_params()
    native = _local(params).eval()
    checkpoint = tmp_path / "native.ckpt"
    torch.save({"state_dict": native.state_dict()}, checkpoint)
    model = SANDMolecularEncoder(**params, batch_size=2, device="cpu")
    assert model.load_from_pretrained(str(checkpoint)) is model
    assert model.is_fitted_ and not model.model.training
    torch.testing.assert_close(model.encode(SMILES), native.encode_smiles(SMILES, batch_size=2))


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_released_dense_checkpoint_matches_upstream(device):
    checkpoint_path = os.environ.get("SAND_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("Set SAND_CHECKPOINT to an official SAND release checkpoint")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    sand = pytest.importorskip("sand")
    config_path = os.environ.get(
        "SAND_CONFIG", str(Path(sand.__file__).parent / "config" / "no_compression.yml")
    )
    native = sand.SANDModel.load_from_checkpoint(
        checkpoint_path, config_path, map_location="cpu"
    ).eval()
    model = SANDMolecularEncoder(
        quantize_emb=False, batch_size=2, device=device
    ).load_from_pretrained(checkpoint_path)
    expected = native.encode_smiles(SMILES, batch_size=2)
    actual = model.encode(SMILES)
    assert actual.shape == expected.shape and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
