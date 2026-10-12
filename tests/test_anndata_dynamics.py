import warnings
import numpy as np
import pandas as pd
import pytest
import torch
import coati as ct
from coati.anndata_io import InputConventionWarning
from coati.result import Result

ad = pytest.importorskip("anndata")


def test_adaptation_preserves_original_order_and_colors():
    x = np.arange(20, dtype=float).reshape(10, 2)
    obs = pd.DataFrame({"stage": ["late"] * 5 + ["early"] * 5,
                        "cell_type": pd.Categorical(["B", "A"] * 5, categories=["B", "A"]),
                        "cell_id": [f"bc-{i}" for i in range(10)]}, index=[f"row-{i}" for i in range(10)])
    original = ad.AnnData(np.zeros((10, 4)), obs=obs)
    original.obsm["X_pca"] = x
    original.obsm["X_umap"] = x / 2
    original.uns["cell_type_colors"] = ["#0072ce", "#f28c28"]
    with pytest.warns(InputConventionWarning, match="COATI recommends"):
        prepared = ct.prepare_anndata(original, time_mapping={"early": 0., "late": 2.})
    assert "celltype" not in original.obs and original.obs_names[0] == "row-0"
    assert prepared.obs_names.tolist() == [f"bc-{i}" for i in range(10)]
    assert prepared.obs["coati_original_obs_name"].tolist() == original.obs_names.tolist()
    np.testing.assert_array_equal(prepared.obs["time"], [2.] * 5 + [0.] * 5)
    np.testing.assert_array_equal(prepared.obsm["X_pca"], x)
    assert prepared.obs["celltype"].cat.categories.tolist() == ["B", "A"]
    assert prepared.uns["celltype_colors"].tolist() == ["#0072ce", "#f28c28"]
    with warnings.catch_warnings(record=True) as caught:
        ct.prepare_anndata(prepared)
    assert not caught
    temporal = ct.TemporalData.from_anndata(prepared)
    np.testing.assert_array_equal(temporal.primary[0], x[5:])
    np.testing.assert_array_equal(temporal.primary[1], x[:5])


def test_arrays_temporal_metadata_and_unknown_labels():
    data = ct.datasets.paired(n=32)
    labels = [np.repeat("type-a", len(x)) for x in data.primary]
    with pytest.warns(InputConventionWarning, match="generated barcodes"):
        a = data.to_anndata(celltype=labels)
    assert a.n_obs == 96 and a.obs_names.is_unique
    np.testing.assert_array_equal(a.obs["time"], np.repeat(data.times, 32))
    np.testing.assert_array_equal(a.obsm["X_secondary"], np.concatenate(data.secondary))
    restored = ct.TemporalData.from_anndata(a, secondary_obsm_key="X_secondary")
    for actual, expected in zip(restored.primary, data.primary):
        np.testing.assert_array_equal(actual, expected)
    with pytest.warns(InputConventionWarning, match="unknown"):
        unknown = ct.prepare_anndata(np.ones((4, 2)), time=0.)
    assert unknown.obs["celltype"].astype(str).tolist() == ["unknown"] * 4
    assert "X_umap" not in unknown.obsm
    uneven = [np.ones((3, 2)), np.ones((5, 2)) * 2]
    a = ct.prepare_anndata(uneven, time=[0., 2.], celltype=[list("AAA"), list("BBBBB")], warn=False)
    np.testing.assert_array_equal(a.obs["time"], [0.] * 3 + [2.] * 5)
    assert a.obs["celltype"].tolist() == list("AAABBBBB")


def test_ambiguous_missing_and_invalid_metadata():
    a = ad.AnnData(np.ones((5, 2)), obs=pd.DataFrame({"stage": [0] * 5, "time_index": [1] * 5}, index=[str(i) for i in range(5)]))
    with pytest.raises(ValueError, match="Multiple candidates"):
        ct.prepare_anndata(a)
    converted = ct.prepare_anndata(a, time_key="stage", warn=False)
    assert converted.obs["time"].tolist() == [0.] * 5
    with pytest.raises(ValueError, match="cannot be inferred"):
        ct.prepare_anndata(np.ones((5, 2)))
    with pytest.raises(ValueError, match="time_mapping"):
        ct.prepare_anndata(np.ones((5, 2)), time=["E1"] * 5)
    with pytest.raises(ValueError, match="unique"):
        ct.prepare_anndata(np.ones((5, 2)), time=0., barcode=["same"] * 5)
    with pytest.raises(ValueError, match="missing"):
        ct.prepare_anndata(np.ones((5, 2)), time=0., celltype=[None] * 5)
    with pytest.raises(ValueError, match="one value per cell"):
        ct.prepare_anndata(np.ones((5, 2)), time=[0., 1.])
    with pytest.raises(ValueError, match="two coordinates"):
        ct.prepare_anndata(np.ones((5, 2)), time=0., umap=np.ones((5, 3)), warn=False)
    with pytest.raises(ValueError, match="counts"):
        ct.prepare_anndata([np.ones((4, 2)), np.ones((5, 2))], time=[0., 1.],
                          secondary=[np.ones((5, 3)), np.ones((4, 3))], warn=False)


@pytest.mark.parametrize("multimodal", [False, True])
def test_anndata_training_preserves_temporal_training(multimodal, tmp_path):
    data = ct.datasets.paired(n=32)
    a = data.to_anndata(celltype="toy", warn=False)
    cfg = ct.Config(multimodal=multimodal, balanced=False, iterations=3, batch_size=26,
                    hidden_dim=8, map_type="kernel", floor_n_sample=26, median_n_sample=32)
    from_ann = ct.train_vector_field(a, cfg, output_dir=tmp_path / "ann")
    from_arrays = ct.train_vector_field(data, cfg, output_dir=tmp_path / "array")
    for key, expected in from_arrays.model.state_dict().items():
        torch.testing.assert_close(from_ann.model.state_dict()[key], expected, rtol=0, atol=0)
    restored = ct.load_result(from_ann.output_dir)
    np.testing.assert_array_equal(restored.compute_growth(), from_ann.compute_growth())


@pytest.fixture
def growth_result(tmp_path):
    from coati._core.Neural import MLPVectorField
    model = MLPVectorField(dim=2, hidden_dim=8, n_layers=0, unbalanced=True, alpha_growth=3.).double()
    with torch.no_grad():
        model.vector_field_net.input_layer.weight.copy_(torch.tensor([[1., 0., 2.], [0., -1., 3.]]))
        model.vector_field_net.input_layer.bias.zero_()
        model.growth_net.input_layer.weight.copy_(torch.tensor([[1., -2., .5]]))
        model.growth_net.input_layer.bias.fill_(.1)
    x = np.random.default_rng(5).normal(size=(60, 2))
    data = ct.TemporalData([x[:30], x[30:]], [0., 1.])
    return Result(model, data, {"dimension": 2, "primary_scale": 2.,
                  "config": {"unbalancedModel": True}}, tmp_path)


def test_growth_matches_saved_ode_and_input_units(growth_result):
    x = np.array([[2., 4.], [-2., 6.], [0., 8.]])
    times = np.array([0., .5, 1.])
    expected = 3 * (.5 * x[:, 0] - x[:, 1] + .5 * times + .1)
    np.testing.assert_allclose(growth_result.compute_growth(x, time=times, chunk_size=1), expected)
    np.testing.assert_allclose(growth_result.compute_growth(x / 2, time=times, raw=False), expected)
    for point, t, rate in zip(x / 2, times, expected):
        z = torch.tensor(point).reshape(1, 2)
        zero = torch.zeros((1, 1), dtype=z.dtype)
        derivative = growth_result.model(torch.tensor(t), (z, zero, zero, zero))[1]
        np.testing.assert_allclose(derivative.detach().numpy().ravel(), [rate])
    with pytest.raises(ValueError, match="numeric time"):
        growth_result.compute_growth(x)
    with pytest.raises(ValueError, match="finite"):
        growth_result.compute_growth(x, time=np.nan)
    with pytest.raises(ValueError, match="chunk_size"):
        growth_result.compute_growth(x, time=0., chunk_size=0)


def test_growth_heatmap_streams_labels_and_storage(growth_result, tmp_path):
    import matplotlib.pyplot as plt
    x = np.concatenate(growth_result.data.primary)
    times = np.repeat([0., 1.], 30)
    order = np.random.default_rng(4).permutation(len(x))
    a = ct.prepare_anndata(x[order], time=times[order], celltype=np.where(x[order, 0] > 0, "B", "A"),
                          barcode=[f"bc-{i}" for i in order], umap=x[order] @ np.array([[2., -.5], [1., 1.]]), warn=False)
    a.uns["celltype_colors"] = ["#0072ce", "#f28c28"]
    axes = growth_result.plot_dynamics(a, ridge=0)
    assert len(axes) == 2
    assert {text.get_text() for text in axes[0].texts} == {"A", "B"}
    assert {text.get_text() for text in axes[1].texts} == {"A", "B"}
    expected = growth_result.compute_growth(x[order], time=times[order])
    np.testing.assert_allclose(axes[1].collections[0].get_array(), expected)
    np.testing.assert_allclose(axes[0].get_xlim(), axes[1].get_xlim())
    assert "coati_growth" not in a.obs and "coati_velocity" not in a.obsm
    axes[0].figure.savefig(tmp_path / "dynamics.png")
    plt.close(axes[0].figure)
    ax = growth_result.plot_growth(a, show_velocity=True, ridge=0)
    assert len(ax.collections) > 1
    plt.close(ax.figure)
    saved = growth_result.write_dynamics(a, copy=True, ridge=0)
    assert saved is not a and "coati_growth" not in a.obs
    np.testing.assert_allclose(saved.obs["coati_growth"], expected)
    saved.write_h5ad(tmp_path / "dynamics.h5ad")
    reread = ad.read_h5ad(tmp_path / "dynamics.h5ad")
    assert reread.obs_names.equals(a.obs_names)
    np.testing.assert_allclose(reread.obs["coati_growth"], expected)
    np.testing.assert_allclose(reread.obsm["coati_velocity_umap"], saved.obsm["coati_velocity_umap"])


def test_balanced_dynamics_has_no_invented_growth(growth_result):
    import matplotlib.pyplot as plt
    growth_result.metadata["config"]["unbalancedModel"] = False
    x = np.concatenate(growth_result.data.primary)
    a = ct.prepare_anndata(x, time=np.repeat([0., 1.], 30), celltype="toy", umap=x, warn=False)
    with pytest.raises(ValueError, match="unbalanced"):
        growth_result.compute_growth()
    with pytest.raises(ValueError, match="balanced=False"):
        growth_result.plot_dynamics(a, include_growth=True)
    axes = growth_result.plot_dynamics(a)
    assert len(axes) == 1
    plt.close(axes[0].figure)
    saved = growth_result.write_dynamics(a, copy=True)
    assert "coati_growth" not in saved.obs


def test_explicit_array_stage_mapping_and_secondary_selection():
    parts = [np.ones((4, 2)), np.ones((6, 2))]
    a = ct.prepare_anndata(parts, time=["early", "late"], time_mapping={"early": 0., "late": 2.},
                           celltype="toy", warn=False)
    np.testing.assert_array_equal(a.obs["time"], [0.] * 4 + [2.] * 6)
    a.obsm["X_secondary"] = np.ones((10, 3))
    a.obsm["X_atac"] = np.ones((10, 4))
    chosen = ct.prepare_anndata(a, secondary_key="X_atac", warn=False)
    assert chosen.uns["coati_input"]["secondary_rep"] == "X_atac"
    assert chosen.uns["coati_input"]["generated_from_arrays"]
