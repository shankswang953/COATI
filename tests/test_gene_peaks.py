import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from coati.downstream import (
    fit_feature_decoder, find_gene_peaks, compute_gene_peak_dynamics,
)


def example():
    # Deliberately shuffled feature columns and link rows exercise identity matching.
    gene = fit_feature_decoder([[0.], [1.]], [[20., 0.], [40., 2.]],
                              features=['Other', 'G'], units='Expression',
                              n_neighbors=1, max_distance=np.inf)
    peaks = fit_feature_decoder([[0.], [1.]], [[2., 0.], [0., 3.]],
                               features=['p2', 'p1'], binary=True, units='Open probability',
                               n_neighbors=1, max_distance=np.inf)
    links = pd.DataFrame({'gene': ['G', 'G', 'G', 'Other'],
                          'peak': ['p2', 'p1', 'p1', 'p3'],
                          'distance': [20, -5, 8, 1],
                          'region_type': ['promoter', 'promoter', 'distal', 'promoter']})
    pred = {'times': [0., 1.], 'primary': np.array([[[0.]], [[1.]]]),
            'secondary': np.array([[[1.]], [[0.]]]), 'log_weights': np.zeros((2, 1))}
    return gene, peaks, links, pred


def test_gene_lookup_is_annotation_based_sorted_unique_and_readonly(tmp_path):
    _, _, links, _ = example()
    before = links.copy(deep=True)
    path = tmp_path / 'links.csv'
    links.to_csv(path, index=False)
    found = find_gene_peaks('G', path, region_type='promoter', max_peaks=1)
    assert found.peak.tolist() == ['p1']
    assert find_gene_peaks('G', links).peak.tolist() == ['p1', 'p2']
    pd.testing.assert_frame_equal(links, before)
    custom = links.rename(columns={'gene': 'symbol', 'peak': 'interval'})
    assert find_gene_peaks('G', custom, gene_key='symbol', peak_key='interval').peak.tolist() == ['p1', 'p2']
    with pytest.raises(ValueError, match='No candidate'):
        find_gene_peaks('g', links)
    with pytest.raises(ValueError, match='No candidate'):
        find_gene_peaks('G', links, region_type='missing')
    with pytest.raises(ValueError, match='positive integer'):
        find_gene_peaks('G', links, max_peaks=0)


def test_gene_pair_keeps_modality_coordinates_and_feature_identity():
    gene, peaks, links, pred = example()
    pair = compute_gene_peak_dynamics('G', pred, gene_decoder=gene, peak_decoder=peaks,
                                      peak_gene_links=links, weighting='growth')
    assert pair.rna.features == ('G',) and pair.atac.features == ('p1', 'p2')
    np.testing.assert_array_equal(pair.rna.values[:, 0, 0], [0, 2])
    np.testing.assert_array_equal(pair.atac.values[:, 0], [[1, 0], [0, 1]])
    assert gene.features == ('Other', 'G') and peaks.features == ('p2', 'p1')
    with pytest.raises(ValueError, match='Missing ATAC'):
        compute_gene_peak_dynamics('Other', pred, gene_decoder=gene, peak_decoder=peaks, peak_gene_links=links)
    with pytest.raises(ValueError, match='matching time and path'):
        compute_gene_peak_dynamics('G', {**pred, 'secondary': pred['secondary'][:, :0]},
                                  gene_decoder=gene, peak_decoder=peaks, peak_gene_links=links)


def test_joint_separate_plots_observed_toggle_and_native_units():
    gene, peaks, links, pred = example()
    pair = compute_gene_peak_dynamics('G', pred, gene_decoder=gene, peak_decoder=peaks, peak_gene_links=links)
    observed = pd.DataFrame({'modality': ['RNA', 'ATAC'], 'feature': ['G', 'p1'],
                             'time': [1., 1.], 'mean': [8., .2], 'cohort': ['Target', 'Target']})
    try:
        hidden = pair.plot(observed=observed, show_iqr=False)
        assert len(hidden['gene']) == 1 and len(hidden['peaks']) == 2
        assert hidden['gene'][0].figure is hidden['peaks'][0].figure
        assert not hidden['gene'][0].collections
        shown = pair.plot(show_observed=True, observed=observed, show_iqr=False)
        np.testing.assert_array_equal(shown['gene'][0].collections[0].get_offsets(), [[1., 8.]])
        np.testing.assert_array_equal(shown['peaks'][0].collections[0].get_offsets(), [[1., .2]])
        assert shown['gene'][0].get_ylabel() == 'Expression'
        assert shown['peaks'][0].get_ylabel() == 'Open probability'
        assert shown['peaks'][0].get_ylim() == (0., 1.)
        separate = pair.plot(layout='separate')
        assert separate['gene'][0].figure is not separate['peaks'][0].figure
        with pytest.raises(ValueError, match='Observed overlays require'):
            pair.plot(show_observed=True)
        with pytest.raises(ValueError, match='layout'):
            pair.plot(layout='overlay')
        with pytest.raises(ValueError, match='unique'):
            pair.plot(show_observed=True, observed=pd.concat([observed, observed]))
    finally:
        plt.close('all')
