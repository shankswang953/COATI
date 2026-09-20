import json

import numpy as np
import pytest
import torch

from coati import Config, MapConfig, TemporalData, datasets, fit, load_result
from coati.maps import MapQualityWarning, PairedMap, load_map, train_map


def small_config(**kwargs):
    return Config(niters=2, num_samples=26, floor_n_sample=26, median_n_sample=32,
                  hidden_dim=8, sync_loss=True, sync_weight=.35, **kwargs)


def quick_map(**kwargs):
    return MapConfig(epochs=4, hidden_dim=16, eval_samples=16, eval_repeats=2, **kwargs)


@pytest.mark.parametrize("kind", ["film", "mlp"])
def test_automatic_map_loso_reload_and_gradients(kind, tmp_path):
    complete = datasets.paired(n=40)
    data = complete.without_times([.5])
    with pytest.warns(MapQualityWarning):
        result = fit(data, small_config(map_type=kind), map_config=quick_map(), output_dir=tmp_path / kind)
    assert result.map_report["training_times"] == [0., 1.]
    assert result.map_report["kind"] == kind
    assert [row["time"] for row in result.map_report["stages"]] == [0., 1.]
    with np.load(result.output_dir / "map/splits.npz") as splits:
        assert set(splits) == {"train_0", "validation_0", "train_1", "validation_1"}
        for k in range(2):
            assert not np.intersect1d(splits[f"train_{k}"], splits[f"validation_{k}"]).size
            assert len(splits[f"train_{k}"]) + len(splits[f"validation_{k}"]) == 40
    query_times = [0., .5, 1.]
    prediction = result.predict(times=query_times)
    secondary = result.project_secondary(prediction)
    assert secondary.shape == (3, 40, 3)
    loaded = load_result(result.output_dir)
    np.testing.assert_array_equal(secondary, loaded.project_secondary(loaded.predict(times=query_times)))
    assert loaded.map_report == result.map_report
    assert not any(p.requires_grad for p in result.T_model.parameters())
    q = torch.randn(2, 5, 2, requires_grad=True)
    result.T_model(q, .5).sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    if kind == "mlp":
        torch.testing.assert_close(result.T_model(q, 0.), result.T_model(q, 1.))
        torch.testing.assert_close(result.T_model(q), result.T_model(q, 1.))
    else:
        with pytest.raises(ValueError, match="requires a time"):
            result.T_model(q)


def test_external_time_independent_map_skips_pretraining(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("A supplied map must not be trained.")
    monkeypatch.setattr("coati.maps.train_map", fail)
    custom = torch.nn.Linear(2, 3)
    initial = {key: value.clone() for key, value in custom.state_dict().items()}
    result = fit(datasets.paired(n=32), small_config(), T_model=custom, output_dir=tmp_path / "external")
    assert not (result.output_dir / "map").exists()
    assert result.map_report is None and result.metadata["custom_map"]
    for key, value in custom.state_dict().items():
        torch.testing.assert_close(value, initial[key])
    assert custom.training and all(p.requires_grad for p in custom.parameters())
    np.testing.assert_array_equal(result.project_secondary(),
                                  load_result(result.output_dir, T_model=custom).project_secondary())
    with pytest.raises(ValueError, match="original T_model"):
        load_result(result.output_dir).project_secondary()


def test_map_training_uses_training_rows_and_coordinates(tmp_path):
    data = datasets.paired(n=40)
    with pytest.warns(MapQualityWarning):
        model, report = train_map(data, kind="film", config=quick_map(), directory=tmp_path,
                                  device="cpu", seed=7, primary_scale=2., secondary_scale=3.)
    with np.load(tmp_path / "splits.npz") as splits:
        train_x = np.concatenate([x[splits[f"train_{i}"]] / 2 for i, x in enumerate(data.primary)])
        train_y = np.concatenate([y[splits[f"train_{i}"]] / 3 for i, y in enumerate(data.secondary)])
    np.testing.assert_allclose(model.input_mean, train_x.mean(0), atol=1e-7)
    np.testing.assert_allclose(model.target_mean, train_y.mean(0), atol=1e-7)
    assert report["primary_scale"] == 2. and report["secondary_scale"] == 3.
    torch.testing.assert_close(model(torch.tensor(data.primary[0] / 2), 0.),
                               load_map(tmp_path / "model.pt")(torch.tensor(data.primary[0] / 2), 0.))


def test_poor_quality_can_stop_before_trajectory_training(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Trajectory fitting must not start after a failed map quality gate.")
    monkeypatch.setattr("coati._core.Training.run_training", fail)
    with pytest.raises(RuntimeError, match="Trajectory fitting was not started"):
        fit(datasets.paired(n=32), small_config(), output_dir=tmp_path / "rejected",
            map_config=quick_map(quality_action="error", max_normalized_mse=1e-10))
    report = json.loads((tmp_path / "rejected/map/report.json").read_text())
    assert report["status"] == "warning" and report["warnings"]
    assert (tmp_path / "rejected/map/model.pt").exists()
    assert not (tmp_path / "rejected/checkpoints/path_param.pth").exists()


def test_map_improves_paired_prediction(tmp_path):
    rng = np.random.default_rng(0)
    x = [rng.normal(size=(160, 2)).astype(np.float32) for _ in range(2)]
    data = TemporalData(x, [0., 1.], [np.column_stack((2*z[:, 0] - z[:, 1], z[:, 0] + z[:, 1])) for z in x])
    model, report = train_map(data, kind="mlp", config=MapConfig(epochs=120, hidden_dim=32,
                            batch_size=128, eval_repeats=2), directory=tmp_path, device="cpu", seed=0)
    assert report["status"] == "passed"
    assert all(row["normalized_mse"] < .1 for row in report["stages"])
    assert report["best_epoch"] <= report["epochs_completed"]


def test_film_learns_time_dependence_and_is_repeatable(tmp_path):
    rng = np.random.default_rng(1)
    x = [rng.normal(size=(128, 2)).astype(np.float32) for _ in range(2)]
    data = TemporalData(x, [2., 4.], [x[0], x[1] + 4.])
    cfg = MapConfig(epochs=150, hidden_dim=32, batch_size=128, eval_repeats=1)
    first, report = train_map(data, kind="film", config=cfg, directory=tmp_path / "first", device="cpu", seed=9)
    second, repeated = train_map(data, kind="film", config=cfg, directory=tmp_path / "second", device="cpu", seed=9)
    assert report == repeated
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    q = torch.zeros(8, 2)
    torch.testing.assert_close(first(q, 4.) - first(q, 2.), torch.full((8, 2), 4.), rtol=.05, atol=.1)


def test_excluded_stage_cannot_affect_default_map(tmp_path):
    original = datasets.paired(n=32)
    changed = datasets.paired(n=32)
    changed.primary[1][:] = 1e5
    changed.secondary[1][:] = -1e5
    with pytest.warns(MapQualityWarning):
        first, report = train_map(original.without_times([.5]), kind="film", config=quick_map(),
                                  directory=tmp_path / "first", device="cpu", seed=0)
    with pytest.warns(MapQualityWarning):
        second, repeated = train_map(changed.without_times([.5]), kind="film", config=quick_map(),
                                     directory=tmp_path / "second", device="cpu", seed=0)
    assert report == repeated
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


def test_normalized_end_to_end_map(tmp_path):
    torch.save({"scale": 2.}, tmp_path / "rna.pt")
    torch.save({"scale": 3.}, tmp_path / "atac.pt")
    with pytest.warns(MapQualityWarning):
        result = fit(datasets.paired(n=32), small_config(), map_config=quick_map(),
                     primal_norm_path=tmp_path / "rna.pt", sec_norm_path=tmp_path / "atac.pt",
                     output_dir=tmp_path / "scaled")
    np.testing.assert_allclose(result.project_secondary(), 3 * result.project_secondary(result.predict(raw=False), raw=False))
    np.testing.assert_array_equal(result.project_secondary(), load_result(result.output_dir).project_secondary())


def test_automatic_map_with_unbalanced_dynamics(tmp_path):
    with pytest.warns(MapQualityWarning):
        result = fit(datasets.paired(n=32), small_config(unbalancedModel=True, alpha_growth=1.,
                     mass_loss=True, mass_coefficient=1.), map_config=quick_map(), output_dir=tmp_path / "growth")
    prediction = result.predict()
    assert np.isfinite(prediction["log_weights"]).all()
    assert result.project_secondary(prediction).shape[:2] == prediction["log_weights"].shape[:2]
    np.testing.assert_array_equal(result.project_secondary(), load_result(result.output_dir).project_secondary())


def test_config_and_exclusion_validation(tmp_path):
    for kwargs in ({"epochs": 0}, {"n_layers": 1}, {"quality_action": "ignore"},
                   {"validation_fraction": 1}, {"lr": float("nan")}):
        with pytest.raises(ValueError):
            MapConfig(**kwargs)
    data = datasets.paired(n=32)
    with pytest.raises(ValueError, match="must be present"):
        data.without_times([.3])
    with pytest.raises(ValueError, match="at least two"):
        data.without_times([0., 1.])
    with pytest.raises(ValueError, match="map_config is only used"):
        fit(data, small_config(map_type="kernel"), map_config=quick_map(), output_dir=tmp_path / "bad")
    with pytest.raises(ValueError, match="only one"):
        fit(data, small_config(), T_model=torch.nn.Linear(2, 3), map_model=torch.nn.Linear(2, 3))
