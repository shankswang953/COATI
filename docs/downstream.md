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

## Enter one gene name

Supply RNA and ATAC AnnData with molecular columns and their corresponding
model representations, plus an existing peak–gene table. Change `gene`; peak IDs
are looked up automatically.

```python
gene = "Tbx6"
links = ds.find_gene_peaks(gene, "peak_gene_links.csv",
                           region_type="promoter", max_peaks=2)

gene_decoder = ds.fit_feature_decoder(rna, features=[gene], use_rep="X_pca",
                                      units="Log-normalized expression")
peak_decoder = ds.fit_feature_decoder(atac, features=links["peak"].tolist(),
                                      use_rep="X_lsi", binary=True,
                                      units="Local peak-open probability")
pair = ds.compute_gene_peak_dynamics(
    gene, trajectory, gene_decoder=gene_decoder, peak_decoder=peak_decoder,
    peak_gene_links=links, selection=lineage, weighting="growth",
)
pair.plot(layout="combined")  # one figure; separate RNA/ATAC panels and units
pair.plot(layout="separate")  # a gene figure and a peak figure
```

Use `weighting="uniform"` for balanced models. The link table needs `gene` and
`peak` columns; custom column names are supported through `gene_key` and
`peak_key`. `region_type=None` includes all candidate regions. Candidates are
ordered by absolute TSS `distance` when supplied, otherwise by table order;
`max_peaks=None` retains all eligible candidates. Selection does not use
reconstructed curves or their quality scores. Missing gene, link or observed
feature IDs raise an error instead of silently skipping them or inventing links.

This assumes the RNA/ATAC coordinates have the same preprocessing and units as
`trajectory["primary"]` and `trajectory["secondary"]`. The readouts search kNN
in the respective latent spaces, not UMAP. Expression and peak probabilities
keep their own axes; they are not placed on an arbitrary common physical scale.

## Optional observed reference points

Observed overlays are **off by default**. Supply measured snapshot means with
columns `modality` (`RNA`/`ATAC`), `feature`, `time`, `mean`, and optionally
`cohort`. These means must come from explicitly chosen observed cohorts, with
peak counts converted to presence before averaging for a binary readout.

```python
pair.plot(layout="combined", show_observed=False)
pair.plot(layout="combined", show_observed=True, observed=observed_summary)
```

The [biological tutorial](tutorials/09_lineage_features.ipynb) builds this table
from observed NMP and Somitic annotations, records cohort cell counts, and lets
you set `SHOW_OBSERVED`. Such snapshots are different cells at different times,
not measured longitudinal changes of the same cells.

## Check local molecular readouts

```python
# Use independent observations, not the fitting rows.
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
observed = ds.read_h5ad_features("rna.h5ad", [gene], obs_names=cell_barcodes)
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

## Lower-level summaries and plots

`compute_feature_dynamics`, `plot_feature_dynamics` and `plot_feature_heatmap`
remain available for arbitrary gene or peak lists. The gene-centered container
exposes `pair.rna`, `pair.atac` and `pair.links` for further analysis.

```python
pair.rna.summary.to_csv("lineage_gene_dynamics.csv", index=False)
pair.atac.summary.to_csv("lineage_peak_dynamics.csv", index=False)
ds.plot_feature_heatmap(pair.atac, standardize=True)
```

Growth weighting normalizes the saved mass weights within supported selected
paths at each time; it does not reconstruct absolute cell counts. IQR is across
interpolated paths, not a molecular confidence interval. Missing neighborhood
support yields NaNs and plot gaps. Inspect coverage and effective-path counts
before interpreting a mean. Row z-scores are optional visualization, not a common
physical scale for expression and accessibility.

Peak-gene pairs must come from explicit annotations or a separately justified
linking analysis. Comparing two interpolated curves does not prove regulation,
chromatin priming or temporal precedence.
