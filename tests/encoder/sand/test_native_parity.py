"""Optional parity with the upstream SAND implementation."""

import copy
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from torch_molecule import SANDMolecularEncoder

from .utils import PAIRS, SCORES, SEED, SMILES, native_model


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
    native = native_model(params).train()
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
    native = native_model(params)
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
