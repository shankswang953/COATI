# Molecular trajectory readouts

Use reusable functions for lineage selection, molecular readouts, summaries and
plots. Keep the initial cohort, terminal annotations, gene/peak list and biological
interpretation explicit in a tutorial. The [gastrulation example](tutorials/09_lineage_features.ipynb)
executes the whole workflow, including endpoint and held-out readout checks.

## Start a cohort at an observed time

```python
import coati as ct
import numpy as np
from coati import downstream as ds

result = ct.load_result("outputs/my-model")
initial = (cells.obs["time"] == 1.) & (cells.obs["celltype"] == "NMP")
trajectory = result.predict_trajectory(cells.obsm["X_pca"][initial],
                                      start_time=1., times=np.linspace(1., 2.5, 16))
trajectory["secondary"] = result.project_secondary(trajectory)
lineage = ds.select_lineage(trajectory, cells, terminal_celltypes="Somitic mesoderm",
                            use_rep="X_pca", min_probability=.6)
lineage.assignments  # includes rejected paths, probabilities and support distances
```

Choose the same features and units used by the model. `start_time` requires an
explicit initial cohort for a later start and keeps the query inside the observed
range. The example uses a 0.1 model-time solver grid, matching the demonstration training step.
Terminal label transfer uses only observations at the final time. The returned
cohort is model-based and annotation-conditioned, not experimental lineage tracing.
An empty selection remains empty and cannot be summarized as that lineage.

## Fit separate RNA and ATAC feature readouts

```python
# Each AnnData contains molecular columns plus its corresponding model coordinates.
gene_decoder = ds.fit_feature_decoder(rna_with_genes, features=["T", "Sox2", "Tbx6"],
                                      use_rep="X_pca", units="Log-normalized expression")
peak_decoder = ds.fit_feature_decoder(atac_with_peaks, features=peak_ids,
                                      use_rep="X_lsi", binary=True,
                                      units="Local peak-open probability")

# Use an independent observation split for this check, not the fitting rows.
gene_decoder.validate(test_rna_coordinates, test_gene_values)
peak_decoder.validate(test_atac_coordinates, test_peak_counts)
```

The functions do not invert PCA/LSI. A local distance-weighted decoder transfers
observed features from the corresponding latent space. Binary peak targets return
local open probabilities, not hard predicted-cell accessibility states. Validation
compares all supplied test cells with a fixed training-mean baseline, including
points outside the empirical support cutoff. For peaks, MSE is a Brier score;
nonpositive skill fails to improve the constant training open fraction.

For large on-disk atlases, selected columns can be read without loading the atlas:

```python
observed = ds.read_h5ad_features("rna.h5ad", ["T", "Sox2"], obs_names=cell_barcodes)
observed.values       # aligned cells × selected features
observed.barcodes     # requested row order, verified against source identities
observed.feature_ids  # resolved source variable IDs

decoder = ds.fit_feature_decoder(model_coordinates, observed.values,
                                 features=observed.features)
```

The reader supports dense, CSR and CSC matrices, source layers, `.raw`, feature
aliases through `feature_key`, explicit barcode alignment and binary presence.
Files are opened read-only and sparse data is streamed in blocks. No normalization
or peak-gene discovery is performed. Choose raw counts, log expression or another
measured layer deliberately and label its units. Reuse prepared coordinate units;
matching dimension counts alone do not establish matching preprocessing.

## Summarize and plot accepted paths

```python
genes = ds.compute_feature_dynamics(gene_decoder, trajectory, selection=lineage,
                                    weighting="growth")
peaks = ds.compute_feature_dynamics(peak_decoder, trajectory, modality="secondary",
                                    selection=lineage, weighting="growth")
ds.plot_feature_dynamics(genes)
ds.plot_feature_dynamics(peaks, color="#e69f00")
ds.plot_feature_heatmap(genes, standardize=True)
genes.summary.to_csv("lineage_gene_dynamics.csv", index=False)
```

Use uniform weights for balanced models. Growth weighting normalizes the saved
mass weights within supported selected paths at each time; it does not reconstruct
absolute cell counts. IQR is across interpolated paths, not a molecular confidence
interval. Missing neighborhood support yields NaNs and plot gaps. Inspect coverage
and effective-path counts before interpreting a mean. Row z-scores are optional
visualization, not a common physical scale for expression and accessibility.

Peak-gene pairs must come from explicit annotations or a separately justified
linking analysis. Comparing two interpolated curves does not prove regulation,
chromatin priming or temporal precedence.
