"""Training loop, optimisation, and progress reporting."""

import math

import numpy as np
import pytest
import torch

from torch_molecule import GenMolMolecularGenerator

from .utils import SMILES, seed, small_model


def test_targets_and_invalid_training_inputs_are_rejected():
    model = small_model()
    with pytest.raises(ValueError):
        model.fit(SMILES, np.arange(len(SMILES), dtype=np.float32))
    with pytest.raises(ValueError):
        model.fit([])
    with pytest.raises(ValueError):
        model.fit(["not-a-molecule"])
    assert not model.is_fitted_


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_updates_weights_with_finite_gradients(device, monkeypatch, genmol_dependencies):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    seed()
    observations = []
    original_step = torch.optim.AdamW.step

    def monitored_step(optimizer, *args, **kwargs):
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        gradients = [p.grad for p in parameters if p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        previous = [p.detach().clone() for p in parameters]
        result = original_step(optimizer, *args, **kwargs)
        assert all(torch.isfinite(p).all() for p in parameters)
        observations.append(
            any(not torch.equal(before, after) for before, after in zip(previous, parameters))
        )
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", monitored_step)
    model = small_model(device=device)
    assert model.fit(SMILES) is model
    assert model.is_fitted_ and len(model.fitting_loss) == model.epochs
    assert np.isfinite(model.fitting_loss).all()
    assert all(observations)
    assert len(observations) == model.epochs * math.ceil(len(SMILES) / model.batch_size)
    assert model.model.ema.num_updates == len(observations)
    assert all(torch.isfinite(value).all() for value in model.model.ema.shadow_params)
    assert len(model.sequence_lengths_) == len(SMILES)


def test_fit_reinitializes_model_and_training_history(genmol_dependencies):
    seed()
    model = small_model(epochs=1)
    model.fit(SMILES)
    previous = model.model
    with torch.no_grad():
        for parameter in previous.parameters():
            parameter.fill_(42)
    model.fit(SMILES[:3])
    assert model.model is not previous
    assert len(model.fitting_loss) == 1
    assert model.model.ema.num_updates == 1
    assert len(model.sequence_lengths_) == 3
    assert all(torch.isfinite(parameter).all() for parameter in model.model.parameters())


def test_training_warns_that_polymer_endpoints_are_not_preserved(genmol_dependencies):
    model = small_model(epochs=1)
    with pytest.warns(UserWarning, match="do not preserve polymer endpoints") as caught:
        assert model.fit(["*CC*", "*CO*", "CCO"]) is model
    assert len(caught) == 1
    assert model.is_fitted_ and np.isfinite(model.fitting_loss).all()


def test_fit_uses_generator_hooks_and_records_batch_loss_means(make_model):
    class ObservedGenerator(GenMolMolecularGenerator):
        def _convert_to_pytorch_data(self, X, y=None):
            self.converted_smiles = list(X)
            return super()._convert_to_pytorch_data(X, y)

        def _train_epoch(self, train_loader, optimizer, epoch, global_pbar=None, scheduler=None):
            losses = super()._train_epoch(train_loader, optimizer, epoch, global_pbar, scheduler)
            self.epoch_results.append((epoch, losses))
            return losses

    model = make_model(model_class=ObservedGenerator)
    model.epoch_results = []
    assert model.fit(SMILES) is model
    assert model.converted_smiles == SMILES
    assert [epoch for epoch, _ in model.epoch_results] == list(range(model.epochs))
    for (_, losses), epoch_loss in zip(model.epoch_results, model.fitting_loss):
        assert isinstance(losses, list)
        assert len(losses) == math.ceil(len(SMILES) / model.batch_size)
        assert all(math.isfinite(loss) for loss in losses)
        assert epoch_loss == pytest.approx(np.mean(losses))
    assert model.fitting_epoch == model.epochs - 1
    assert model.is_fitted_


@pytest.mark.parametrize("verbose", ["progress_bar", "print_statement"])
def test_training_reports_every_batch(make_model, progress_bars, capsys, verbose):
    model = make_model(verbose=verbose)
    model.fit(SMILES)
    total_steps = model.epochs * math.ceil(len(SMILES) / model.batch_size)
    if verbose == "progress_bar":
        assert len(progress_bars) == 1
        bar = progress_bars[0]
        assert bar.total == bar.updates == total_steps
        assert bar.closed
        assert len([log for log in bar.logs if "Step" in log]) == total_steps
    else:
        assert progress_bars == []
        step_logs = [line for line in capsys.readouterr().out.splitlines() if "'Step':" in line]
        assert len(step_logs) == total_steps
        assert all("'Epoch':" in line and "'Loss':" in line for line in step_logs)


def test_training_failure_closes_progress_and_keeps_model_unfitted(
    make_model, progress_bars, monkeypatch
):
    model = make_model(verbose="progress_bar")

    def fail_epoch(*args, **kwargs):
        raise RuntimeError("training interrupted")

    monkeypatch.setattr(model, "_train_epoch", fail_epoch)
    with pytest.raises(RuntimeError, match="training interrupted"):
        model.fit(SMILES)
    assert len(progress_bars) == 1 and progress_bars[0].closed
    assert not model.is_fitted_
