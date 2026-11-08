import json
from pathlib import Path

import numpy as np
import torch

from coati import Config

from gastrulation.film import load_film_checkpoint, load_scale, load_snapshots
from gastrulation.pretrained import (
    load_gastrulation_result,
    load_gastrulation_umap,
    obtain_gastrulation_assets,
)


def test_archived_gastrulation_networks_load_and_infer():
    assets, manifest = obtain_gastrulation_assets()
    primary = assets["rna_pca_by_time.npz"]
    balanced = load_gastrulation_result(
        assets["balanced_cy0p3_s1_iter20000.pth"], primary, manifest=manifest,
    )
    unbalanced = load_gastrulation_result(
        assets["unbalanced_cy0p3_s0_iter20000.pth"], primary, manifest=manifest,
    )
    initial = balanced.data.primary[0][:4]
    times = [0.0, 1.0, 2.5]
    balanced_prediction = balanced.predict_trajectory(initial, times=times)
    unbalanced_prediction = unbalanced.predict_trajectory(initial, times=times)
    assert balanced_prediction["primary"].shape == (3, 4, 50)
    assert unbalanced_prediction["primary"].shape == (3, 4, 50)
    assert "log_weights" not in balanced_prediction
    assert np.isfinite(unbalanced_prediction["log_weights"]).all()
    assert balanced.metadata["checkpoint_iteration"] == 20000
    assert unbalanced.metadata["checkpoint_iteration"] == 20000


def test_original_gastrulation_umap_matches_archived_rna_rows():
    assets, _ = obtain_gastrulation_assets()
    reducer = load_gastrulation_umap(
        assets["rna_umap_model.joblib"], assets["rna_pca_by_time.npz"],
    )
    assert reducer._raw_data.shape == (35503, 50)
    assert reducer.embedding_.shape == (35503, 2)


def test_full_data_training_configs_match_trainf_film_recipes():
    example_dir = Path(__file__).parents[1] / "examples/gastrulation"
    balanced = Config(**json.loads((example_dir / "balanced.json").read_text())["training"]).to_dict()
    unbalanced = Config(**json.loads((example_dir / "unbalanced.json").read_text())["training"]).to_dict()

    assert (balanced["niters"], balanced["num_samples"], balanced["hidden_dim"],
            balanced["n_layers"], balanced["seed"]) == (50000, 1024, 800, 3, 0)
    assert (balanced["sync_weight"], balanced["pdf_coefficient"],
            balanced["energy_coefficient"], balanced["density_coefficient"]) == (.5, 100., .1, .1)
    assert not balanced["unbalancedModel"]
    assert balanced["map_type"] == "FiLMMLP"
    assert balanced["pdf_coefficient_pri"] == 50.

    assert (unbalanced["niters"], unbalanced["num_samples"], unbalanced["hidden_dim"],
            unbalanced["n_layers"], unbalanced["seed"]) == (20000, 1024, 800, 3, 0)
    assert (unbalanced["sync_weight"], unbalanced["pdf_coefficient"],
            unbalanced["energy_coefficient"], unbalanced["density_coefficient"]) == (.5, 100., .01, .01)
    assert unbalanced["unbalancedModel"]
    assert unbalanced["biological_num"] == [1., 4., 6., 11.]


def test_compact_gastrulation_data_normalization_and_film_checkpoint():
    data_dir = Path(__file__).parents[1] / "examples/gastrulation/data"
    import anndata as ad

    cells = ad.read_h5ad(data_dir / "gastrulation_embeddings.h5ad")
    assert cells.shape == (35503, 50)
    assert cells.obsm["X_pca"].shape == (35503, 50)
    assert cells.obsm["X_atac"].shape == (35503, 14)
    assert cells.obs.groupby("stage", observed=True).size().to_dict() == {
        "E7.5": 9018, "E8.0": 7796, "E8.5": 11470, "E8.75": 7219,
    }
    assert load_scale(data_dir / "primal_norm_params.pt") == 22.863716719648387
    assert load_scale(data_dir / "secondary_norm_params.pt") == 0.0015835087817000948
    primary, secondary = load_snapshots(
        data_dir / "gastrulation_embeddings.h5ad",
        data_dir / "primal_norm_params.pt",
        data_dir / "secondary_norm_params.pt",
    )
    torch.testing.assert_close(primary[0][0], torch.tensor(cells.obsm["X_pca"][0]) / 22.863716719648387)
    torch.testing.assert_close(secondary[0][0], torch.tensor(cells.obsm["X_atac"][0]) / 0.0015835087817000948)
    model = load_film_checkpoint(data_dir / "T_FiLM.pt")
    prediction = model(primary[0][:3], 0.0)
    assert prediction.shape == (3, 14)
    assert torch.isfinite(prediction).all()
