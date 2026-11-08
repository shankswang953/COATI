import numpy as np
import torch
from synthetic.gaussian_fresh import generate_fresh_anndata
from synthetic.gaussian_map import fit_bump_map, evaluate_bump_map
from synthetic.gaussian_comparison import compare_gaussian_runs


def test_fresh_anndata_contains_isotropic_gaussian_snapshots():
    cells, pairs = generate_fresh_anndata()
    assert cells.n_obs == 3000 and pairs.n_obs == 10000
    assert cells.obs_names.is_unique and pairs.obs_names.is_unique
    np.testing.assert_array_equal(cells.obs_names.to_numpy(), cells.obs['barcode'].to_numpy())
    assert set(cells.obs.columns) >= {'time', 'celltype', 'barcode'}
    assert set(pairs.obs.columns) >= {'source', 'barcode'}
    np.testing.assert_array_equal(pairs.X[:3000], cells.obsm['X_latent'])
    assert np.count_nonzero(cells.obsm['X_secondary'][:, 2]) == 0
    groups = [cells.obsm['X_latent'][:1000], cells.obsm['X_latent'][1000:2000],
              cells.obsm['X_latent'][2000:]]
    for group in groups:
        covariance = np.cov(group, rowvar=False)
        assert abs(covariance[0, 1]) < 5e-4
        assert abs(covariance[0, 0] - covariance[1, 1]) < 6e-4


def test_gaussian_generator_uses_exact_isotropic_covariances(monkeypatch):
    from synthetic import generate_gaussian_data as generator
    covariances = []
    def capture_covariance(rng, mean, cov, n, low=0., high=1.):
        covariances.append(np.asarray(cov))
        return np.repeat(np.asarray(mean)[None], n, axis=0)
    monkeypatch.setattr(generator, 'sample_truncated_gaussian', capture_covariance)
    generator.make_data()
    np.testing.assert_array_equal(covariances[0], .004 * np.eye(2))
    np.testing.assert_array_equal(covariances[1], .003 * np.eye(2))
    np.testing.assert_array_equal(covariances[2], .003 * np.eye(2))


def test_map_training_uses_fresh_inputs_and_preserves_input_gradients(monkeypatch):
    def reject_load(*args, **kwargs):
        raise AssertionError('The fresh workflow must not read data or pretrained checkpoints.')
    with monkeypatch.context() as patch:
        patch.setattr(np, 'load', reject_load)
        patch.setattr(torch, 'load', reject_load)
        cells, pairs = generate_fresh_anndata()
        model, history, info = fit_bump_map(pairs, steps=2, eval_every=1)
        scores, grid = evaluate_bump_map(model, pairs, grid_n=25)
    assert info['n_training'] == 8500 and info['n_validation'] == 1500
    assert set(pairs.obs['map_split']) == {'Training', 'Validation'}
    assert scores.loc[['Held-out observed', 'Held-out support'], 'Pairs'].sum() == 1500
    assert scores.loc['Independent grid', 'Pairs'] == 625
    assert np.isfinite(scores.to_numpy()).all()
    assert np.isfinite(grid['prediction']).all()
    assert not any(parameter.requires_grad for parameter in model.parameters())
    query = torch.tensor([[.45, .48]], requires_grad=True)
    model(query).sum().backward()
    assert torch.isfinite(query.grad).all() and torch.linalg.norm(query.grad) > 0


def test_balanced_mass_target_does_not_follow_observation_counts():
    from types import SimpleNamespace
    source = np.array([[0., 0.], [0., 1.]])
    primary = np.repeat(source[None], 3, axis=0)
    prediction = dict(times=np.array([0., .5, 1.]), primary=primary,
        secondary=np.concatenate([primary, np.zeros((3, 2, 1))], axis=2), energy=np.zeros((3, 2, 1)))
    result = SimpleNamespace(data=SimpleNamespace(primary=[source, np.repeat(source, 2, axis=0)]),
        metadata={'primary_scale': 1., 'blur_primary': [.05, .05]})
    scores = compare_gaussian_runs({'Balanced': result}, {'Balanced': prediction})
    assert scores.loc['Balanced', 'Predicted terminal mass'] == 1.
    assert scores.loc['Balanced', 'Target terminal mass'] == 1.



def test_shared_assets_match_the_reproducible_generator():
    from synthetic.gaussian_assets import load_shared_gaussian_assets, resolve_asset_directory
    import anndata as ad

    expected_cells, expected_pairs = generate_fresh_anndata()
    cells, _, _ = load_shared_gaussian_assets()
    pairs = ad.read_h5ad(resolve_asset_directory() / 'map_pairs.h5ad')
    for key in ['X_latent', 'X_secondary']:
        np.testing.assert_array_equal(cells.obsm[key], expected_cells.obsm[key])
    np.testing.assert_array_equal(pairs.X, expected_pairs.X)
    np.testing.assert_array_equal(pairs.obsm['X_secondary'], expected_pairs.obsm['X_secondary'])
