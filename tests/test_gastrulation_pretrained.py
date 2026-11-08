import json
from pathlib import Path

import numpy as np

from coati import Config

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


def test_full_data_training_configs_match_archived_models():
    example_dir = Path(__file__).parents[1] / "examples/gastrulation"
    balanced = Config(**json.loads((example_dir / "balanced.json").read_text())["training"]).to_dict()
    unbalanced = Config(**json.loads((example_dir / "unbalanced.json").read_text())["training"]).to_dict()

    assert (balanced["niters"], balanced["num_samples"], balanced["hidden_dim"],
            balanced["n_layers"], balanced["seed"]) == (20000, 256, 400, 4, 1)
    assert (balanced["sync_weight"], balanced["pdf_coefficient"],
            balanced["energy_coefficient"], balanced["density_coefficient"]) == (.3, 100., .1, .1)
    assert not balanced["unbalancedModel"]

    assert (unbalanced["niters"], unbalanced["num_samples"], unbalanced["hidden_dim"],
            unbalanced["n_layers"], unbalanced["seed"]) == (20000, 1024, 800, 3, 0)
    assert (unbalanced["sync_weight"], unbalanced["pdf_coefficient"],
            unbalanced["energy_coefficient"], unbalanced["density_coefficient"]) == (.3, 100., .01, .01)
    assert unbalanced["unbalancedModel"]
    assert unbalanced["biological_num"] == [1., 4., 6., 11.]
