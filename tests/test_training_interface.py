"""Public-device selection and training I/O regressions; no scientific defaults change."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coati import Config, TemporalData, fit, load_result, resolve_device


def data():
    rng = np.random.default_rng(3)
    return TemporalData([rng.normal(0, .05, (32, 2)),
                         rng.normal(0, .05, (32, 2)) + .2], [0., 1.])


def config(**kwargs):
    return Config(iterations=1, batch_size=26, median_n_sample=32,
                  floor_n_sample=26, hidden_dim=8, **kwargs)


def test_explicit_cpu_does_not_probe_accelerators(monkeypatch):
    def unexpected():
        pytest.fail("Explicit CPU selection must not probe accelerators.")
    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(torch.backends.mps, "is_available", unexpected)
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_preserves_original_device_priority(monkeypatch):
    from coati._core.Training import _resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert str(resolve_device("auto")) == "cuda:0"
    assert str(_resolve_device(SimpleNamespace(device="auto", gpu=1))) == "cuda:1"
    assert str(_resolve_device(SimpleNamespace(device="auto", gpu=-1))) == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_device("auto")) == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert str(resolve_device("auto")) == "cpu"
    assert Config().to_dict()["device"] == "cpu"


def test_cuda_indices_are_checked_before_training(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert str(resolve_device("cuda:1")) == "cuda:1"
    with pytest.raises(ValueError, match="only 2 CUDA"):
        resolve_device("cuda:2")


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_unavailable_devices_do_not_create_partial_runs(device, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    output = tmp_path / "unavailable"
    with pytest.raises(RuntimeError, match="unavailable"):
        fit(data(), config(), device=device, output_dir=output)
    assert not output.exists()
    with pytest.raises(RuntimeError, match="unavailable"):
        load_result(tmp_path / "not-loaded", device=device)


@pytest.mark.parametrize("device", ["gpu", "meta", "cuda:invalid"])
def test_invalid_device_names_are_actionable(device):
    with pytest.raises(ValueError, match="device"):
        resolve_device(device)


@pytest.mark.parametrize("settings", [{"lr": float("nan")}, {"dt": float("inf")}, {"niters": True}])
def test_nonfinite_or_boolean_training_parameters_are_rejected(settings):
    with pytest.raises(ValueError, match="positive"):
        Config(**settings)


def test_density_neighborhood_is_checked_without_silent_adjustment(tmp_path):
    cfg = Config(iterations=1, batch_size=16)
    with pytest.raises(ValueError, match="density_topk"):
        fit(data(), cfg, output_dir=tmp_path / "invalid")
    assert cfg.to_dict()["num_samples"] == 16
    assert cfg.to_dict()["density_topk"] == 25
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("verbose", [False, True])
def test_training_logs_completion_paths_and_portable_reload(verbose, tmp_path, capsys):
    cfg = config(device="auto")
    result = fit(data(), cfg, device=torch.device("cpu"),
                 output_dir=tmp_path / "run with spaces", verbose=verbose)
    output = capsys.readouterr().out
    assert cfg.to_dict()["device"] == "auto"
    assert result.metadata["config"]["device"] == "cpu"
    assert result.metadata["resolved_device"] == "cpu"
    assert str(result.output_dir.resolve()) in output
    assert "Training complete" in output
    assert str(result.output_dir / "checkpoints/path_param.pth") in output
    assert "Training complete after 1 iters" in (result.output_dir / "training.log").read_text()
    prediction = result.predict()
    np.testing.assert_array_equal(prediction["primary"], load_result(result.output_dir).predict()["primary"])


@pytest.mark.parametrize("verbose", [False, True])
def test_training_failure_preserves_exception_and_diagnostics(verbose, tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        print("progress before failure")
        print("diagnostic before failure", file=sys.stderr)
        raise RuntimeError("original failure")
    monkeypatch.setattr("coati._core.Training.run_training", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="original failure"):
        fit(data(), config(), output_dir=output, verbose=verbose)
    captured = capsys.readouterr()
    assert "Training complete" not in captured.out
    assert str(output.resolve() / "training.log") in captured.err
    log = (output / "training.log").read_text()
    assert "progress before failure" in log
    assert "diagnostic before failure" in log
    assert "Traceback" in log and "original failure" in log
    assert not (output / "result.json").exists()


def test_cli_device_override_runs_the_public_workflow(tmp_path, capsys):
    from coati.cli import main
    keys = data().save_npz(tmp_path / "inputs")
    cfg = tmp_path / "config.json"
    config(device="cuda:99").save(cfg)
    output = tmp_path / "cli"
    main(["fit", str(tmp_path / "inputs/primary.npz"), "--keys", *keys,
          "--times", "0", "1", "--config", str(cfg), "--device", "cpu",
          "--output", str(output)])
    assert json.loads((output / "config.json").read_text())["device"] == "cpu"
    assert (output / "checkpoints/path_param.pth").exists()
    assert (output / "metrics.json").exists()
    assert (output / "trajectories.png").exists()
    assert str(output) in capsys.readouterr().out
