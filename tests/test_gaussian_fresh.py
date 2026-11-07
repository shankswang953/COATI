import numpy as np
import torch
from gaussian_fresh import generate_fresh_anndata
from gaussian_map import fit_bump_map, evaluate_bump_map
from gaussian_reference import _asset
from gaussian_comparison import compare_gaussian_runs


def test_fresh_anndata_reproduces_the_original_arrays():
    cells, pairs = generate_fresh_anndata()
    assert cells.n_obs == 3000 and pairs.n_obs == 10000
    assert cells.obs_names.is_unique and pairs.obs_names.is_unique
    np.testing.assert_array_equal(cells.obs_names.to_numpy(), cells.obs['barcode'].to_numpy())
    assert set(cells.obs.columns) >= {'time', 'celltype', 'barcode'}
    assert set(pairs.obs.columns) >= {'source', 'barcode'}
    for filename, representation in [('primary.npz', 'X_latent'), ('secondary.npz', 'X_secondary')]:
        with np.load(_asset(filename), allow_pickle=False) as saved:
            np.testing.assert_allclose(cells.obsm[representation], np.concatenate([saved['time0'], saved['time1']]), rtol=1e-6, atol=1e-7)
    with np.load(_asset('map_support.npz'), allow_pickle=False) as saved:
        np.testing.assert_allclose(pairs.X[3000:], saved['support2d'], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(pairs.obsm['X_secondary'][3000:], saved['support3d'], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(pairs.X[:3000], cells.obsm['X_latent'])
    assert np.count_nonzero(cells.obsm['X_secondary'][:, 2]) == 0


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



def test_explicit_notebook_preparation_matches_the_script(monkeypatch):
    import json
    from pathlib import Path
    import anndata as ad
    import pandas as pd
    import coati as ct
    from gaussian_parameters import FLOW_PARAMETERS
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/01_gaussian.ipynb').read_text())
    scope = {'np': np, 'pd': pd, 'ad': ad, 'ct': ct}
    def reject_load(*args, **kwargs):
        raise AssertionError('The notebook must generate its inputs directly.')
    with monkeypatch.context() as patch:
        patch.setattr(np, 'load', reject_load)
        patch.setattr(torch, 'load', reject_load)
        for index in [3, 5, 14]:
            exec(compile(''.join(notebook['cells'][index]['source']), '<explicit-notebook-cell>', 'exec'), scope)
        expected_cells, expected_pairs = generate_fresh_anndata()
    for key in ['X_latent', 'X_secondary']:
        np.testing.assert_array_equal(scope['cells'].obsm[key], expected_cells.obsm[key])
    np.testing.assert_array_equal(scope['map_pairs'].X, expected_pairs.X)
    np.testing.assert_array_equal(scope['map_pairs'].obsm['X_secondary'], expected_pairs.obsm['X_secondary'])
    pd.testing.assert_frame_equal(scope['cells'].obs, expected_cells.obs)
    pd.testing.assert_frame_equal(scope['map_pairs'].obs, expected_pairs.obs)
    assert scope['FLOW_PARAMETERS'] == FLOW_PARAMETERS
