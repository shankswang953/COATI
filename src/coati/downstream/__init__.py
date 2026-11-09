"""Reusable downstream analyses and molecular readouts for COATI trajectories."""
from .features import (
    FeatureMatrix, FeatureDecoder, FeatureDynamics, LineageSelection,
    read_h5ad_features, fit_feature_decoder, select_lineage,
    compute_feature_dynamics, plot_feature_dynamics, plot_feature_heatmap,
)

from .gene_peaks import (
    GenePeakDynamics, find_gene_peaks, compute_gene_peak_dynamics,
    plot_gene_peak_dynamics,
)

from .mass import MassDecomposition, decompose_mass, plot_mass_decomposition

from .distribution_metrics import (
    DistributionMetrics,
    composition_similarity,
    evaluate_distribution,
    evaluate_paired_distributions,
    normalize_particle_weights,
    sinkhorn_w2,
    support_recall,
)
