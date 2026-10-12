import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse
from coati.downstream import (
    read_h5ad_features, fit_feature_decoder, select_lineage,
    compute_feature_dynamics, plot_feature_dynamics, plot_feature_heatmap,
)


@pytest.mark.parametrize('kind', ['dense','csr','csc'])
def test_feature_file_reorders_cells_and_features(tmp_path, kind):
    values = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]], dtype=float)
    matrix = values if kind == 'dense' else getattr(sparse, kind+'_matrix')(values)
    a = ad.AnnData(matrix, obs=pd.DataFrame(index=['c0','c1','c2']),
                   var=pd.DataFrame({'symbol':['A','B','C']}, index=['id0','id1','id2']))
    a.layers['counts'] = matrix.copy()
    p = tmp_path/'features.h5ad';a.write_h5ad(p)
    got = read_h5ad_features(p,['C','A'],feature_key='symbol',obs_names=['c2','c0'],layer='counts',block_size=1)
    np.testing.assert_array_equal(got.values, values[[2,0]][:,[2,0]])
    assert list(got.barcodes) == ['c2','c0'] and got.feature_ids == ('id2','id0')
    binary = read_h5ad_features(p,['B'],feature_key='symbol',binary=True)
    np.testing.assert_array_equal(binary.values[:,0], [1,0,1])
    with pytest.raises(ValueError,match='Missing source barcodes'):
        read_h5ad_features(p,['id0'],obs_names=['missing'])
    with pytest.raises(ValueError,match='matches'):
        read_h5ad_features(p,['missing'])
    with pytest.raises(ValueError,match='unique'):
        read_h5ad_features(p,['id0'],obs_names=['c0','c0'])


def test_interpolation_exact_matches_duplicates_and_binary_probability():
    decoder = fit_feature_decoder(np.array([[0.],[0.],[1.]]),np.array([[0.],[2.],[0.]]),
                                  features=['peak'],binary=True,n_neighbors=3,max_distance=np.inf)
    values,dist,supported = decoder.predict(np.array([[[0.],[.5]]]))
    np.testing.assert_allclose(values[0,:,0], [.5,1/3])
    assert supported.all() and values.min()>=0 and values.max()<=1


def test_support_masks_growth_weights_and_missing_lineage():
    decoder = fit_feature_decoder([[0.],[1.]],[[0.],[2.]],features=['gene'],max_distance=.2,n_neighbors=2)
    pred={'times':np.array([0.,.5,1.]), 'primary':np.array([[[0.],[1.]],[[0.],[1.]],[[4.],[5.]]]),
          'log_weights':np.array([[[np.log(.25)],[np.log(.75)]],[[0.],[-1000.]],[[0.],[0.]]])}
    uniform=compute_feature_dynamics(decoder,pred)
    weighted=compute_feature_dynamics(decoder,pred,weighting='growth')
    np.testing.assert_allclose(uniform.summary['mean'][:2], [1,1])
    np.testing.assert_allclose(weighted.summary['mean'][:2], [1.5,0])
    assert weighted.summary.q25.iloc[1]==0 and weighted.summary.q75.iloc[1]==0
    assert np.isnan(weighted.summary['mean'].iloc[2])
    assert weighted.summary.coverage.tolist()==[1.,1.,0.]
    assert np.all(weighted.summary.effective_paths <= 2)
    with pytest.raises(ValueError,match='log_weights'):
        compute_feature_dynamics(decoder,{'times':pred['times'],'primary':pred['primary']},weighting='growth')
    for axes in [plot_feature_dynamics(uniform)]:
        assert np.isnan(axes[0].lines[0].get_ydata()[-1])
    plot_feature_heatmap(uniform)


def test_lineage_terminal_reference_and_empty_selection():
    obs=pd.DataFrame({'time':[0.,0.,1.,1.], 'celltype':['B','B','A','B']},index=['a','b','c','d'])
    a=ad.AnnData(np.array([[0.],[1.],[0.],[1.]]),obs=obs)
    a.obsm['X_pca']=np.asarray(a.X).copy()
    pred={'times':[0.,1.], 'primary':np.array([[[0.],[1.],[4.]],[[0.],[1.],[4.]]])}
    selected=select_lineage(pred,a,terminal_celltypes=['A'],n_neighbors=1,max_distance=.2)
    assert selected.indices.tolist()==[0]
    assert selected.assignments.supported.tolist()==[True,True,False]
    empty=select_lineage({'times':[0.,1.],'primary':pred['primary'][:,1:2]},a,terminal_celltypes=['A'],n_neighbors=1,max_distance=.2)
    assert len(empty.indices)==0
    decoder=fit_feature_decoder([[0.],[1.]],[[0.],[2.]],features=['gene'])
    with pytest.raises(ValueError,match='No paths passed'):
        compute_feature_dynamics(decoder,{'times':[0.,1.],'primary':pred['primary'][:,1:2]},selection=empty)


def test_anndata_decoder_keeps_representation_and_feature_order():
    a=ad.AnnData(np.array([[2.,5.],[3.,6.],[4.,7.]]),var=pd.DataFrame(index=['A','B']))
    a.obsm['X_pca']=np.array([[0.],[1.],[2.]])
    d=fit_feature_decoder(a,features=['B','A'],n_neighbors=1)
    values,_,_=d.predict([[1.]])
    np.testing.assert_array_equal(values, [[6.,3.]])
    np.testing.assert_array_equal(a.X, [[2.,5.],[3.,6.],[4.,7.]])


def test_decoder_validation_uses_frozen_training_baseline_and_does_not_hide_unsupported():
    d=fit_feature_decoder([[0.],[1.]],[[0.],[1.]],features=['peak'],binary=True,max_distance=.1,n_neighbors=1)
    result=d.validate([[0.],[1.],[4.]],[[0.],[1.],[0.]])
    assert result.test_cells.iloc[0]==3
    assert result.support_coverage.iloc[0]==pytest.approx(2/3)
    assert result.Brier.iloc[0]==pytest.approx(1/3)
    assert result.skill_vs_training_mean.iloc[0]<0
