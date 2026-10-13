import numpy as np
import pandas as pd
import pytest
import anndata as ad
from coati.downstream import decompose_mass


def example():
    reference = ad.AnnData(np.array([[0.], [10.], [0.], [10.]]),
        obs=pd.DataFrame({'time':[0.,0.,1.,1.], 'celltype':['NMP','Other','NMP','Other']}, index=list('abcd')))
    reference.obsm['X_latent'] = reference.X.copy()
    # Retained, incoming, outgoing, unrelated: preserve every particle's identity.
    prediction = dict(times=[0.,1.], primary=np.array([[[0.],[10.],[0.],[10.]], [[0.],[0.],[10.],[10.]]]),
        log_weights=np.log(np.array([[.2,.1,.3,.4], [.25,.3,.2,.25]]))[...,None])
    return prediction, reference


def test_paper_identity_uses_later_incoming_and_earlier_outgoing():
    pred, ref = example()
    original = ref.copy()
    result = decompose_mass(pred, ref, celltype='NMP', use_rep='X_latent')
    row = result.summary.loc['NMP']
    np.testing.assert_allclose(row[['incoming','outgoing','reweighting','net_change']].to_numpy(dtype=float), [.3,.3,.05,.05])
    assert abs(row.closure_error) < 1e-14
    assert result.assignments.component.tolist() == ['reweighting','incoming','outgoing','other']
    assert result.assignments.particle.tolist() == [0,1,2,3]
    assert row.observed_change == 0
    np.testing.assert_allclose(result.assignments.start_weight.sum(), 1)
    np.testing.assert_allclose(result.transitions.end_mass.sum(), 1)
    np.testing.assert_array_equal(ref.X, original.X)
    pd.testing.assert_frame_equal(ref.obs, original.obs)
    # Global shifts of log mass cannot change a relative-proportion decomposition.
    pred['log_weights'][0] += 10000
    pred['log_weights'][1] -= 10000
    shifted = decompose_mass(pred, ref, celltype='NMP', use_rep='X_latent')
    np.testing.assert_allclose(shifted.summary, result.summary, atol=1e-12)


def test_balanced_mass_and_reference_order_do_not_change_transitions():
    pred, ref = example()
    pred.pop('log_weights')
    got = decompose_mass(pred, ref, celltype='NMP', use_rep='X_latent')
    row = got.summary.iloc[0]
    assert row.incoming == row.outgoing == .25
    assert row.reweighting == row.net_change == 0
    reordered = decompose_mass(pred, ref[[3,0,2,1]], celltype='NMP', use_rep='X_latent')
    pd.testing.assert_frame_equal(got.summary, reordered.summary)
    # An observed type may be absent from the predicted endpoints; no fabricated mass.
    pred['primary'][:] = 10
    absent = decompose_mass(pred, ref, celltype='NMP', use_rep='X_latent')
    assert (absent.summary[['incoming','outgoing','reweighting','net_change']] == 0).all().all()


def test_interval_and_stage_mapping_are_explicit():
    pred, ref = example()
    pred['times'] = [0.,.5,1.]
    pred['primary'] = np.stack([pred['primary'][0], pred['primary'][0], pred['primary'][1]])
    pred['log_weights'] = np.stack([pred['log_weights'][0], pred['log_weights'][0], pred['log_weights'][1]])
    ref.obs['stage'] = ['early','early','late','late']
    got = decompose_mass(pred, ref, celltype='NMP', time_key='stage', time_mapping={'early':0.,'late':1.}, use_rep='X_latent')
    assert got.summary.net_change.iloc[0] == pytest.approx(.05)
    for kwargs, message in [({'start_time':.2}, 'trajectory time'), ({'start_time':.5}, 'observed reference'),
                             ({'start_time':1.,'end_time':0.}, 'precede'), ({'celltype':'Missing'}, 'absent')]:
        with pytest.raises(ValueError, match=message):
            decompose_mass(pred, ref, **({'celltype':'NMP','use_rep':'X_latent'} | kwargs))
    pred['log_weights'] = pred['log_weights'][:, :2]
    with pytest.raises(ValueError, match='full time-by-particle'):
        decompose_mass(pred, ref, celltype='NMP', use_rep='X_latent')


def test_retained_reweighting_and_plot_signs():
    pred, ref = example()
    pred['primary'][1] = pred['primary'][0]
    got = decompose_mass(pred, ref, celltype='NMP', use_rep='X_latent', n_neighbors=2)
    row = got.summary.iloc[0]
    assert row.incoming == row.outgoing == 0
    assert row.reweighting == pytest.approx(-.05)
    assert row.net_change == pytest.approx(-.05)
    pred, ref = example()
    ax = decompose_mass(pred, ref, celltype='NMP', use_rep='X_latent').plot(show_observed=True)
    np.testing.assert_allclose([patch.get_width() for patch in ax.patches], [5,-30,30])
    assert len(ax.collections) == 2
    import matplotlib.pyplot as plt
    plt.close(ax.figure)
