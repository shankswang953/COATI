import numpy as np
import torch

from synthetic_gaussian_assets import load_shared_gaussian_assets


def test_committed_synthetic_assets_share_data_and_frozen_map():
    cells, model, payload = load_shared_gaussian_assets()
    assert cells.n_obs == 3000
    assert list(cells.obs["time"].value_counts(sort=False)) == [1000, 2000]
    assert cells.obsm["X_latent"].shape == (3000, 2)
    assert cells.obsm["X_secondary"].shape == (3000, 3)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert payload["fit"]["best_step"] == 1500
    with torch.no_grad():
        center = model(torch.tensor([[.5, .5]], dtype=torch.float32)).numpy()
    assert np.isfinite(center).all()
    np.testing.assert_allclose(center[0, :2], [.5, .5], atol=2e-3)
    np.testing.assert_allclose(center[0, 2], 1., atol=2e-2)
