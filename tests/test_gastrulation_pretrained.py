import numpy as np

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
