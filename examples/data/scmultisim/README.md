# scMultiSim paired embeddings

These files are the prepared 5-dimensional RNA and ATAC PCA representations used by
`TraInf/Synthetic/data/MLPVisual.ipynb`. The paired row order and four snapshot sizes
(2,006; 2,001; 3,008; 2,985) are unchanged from the R-generated scMultiSim dataset.

- `all_time_scRNA_pca.npz`: RNA PCA coordinates under keys `time1` to `time4`.
- `all_time_scATAC_pca.npz`: paired ATAC PCA coordinates under the same keys.
- `all_time_celltype.npz`: matching scMultiSim population labels stored as plain
  Unicode arrays; loading never requires pickle.
- `scmultisim.h5ad`: the assembled paired AnnData used by the tutorial, with RNA PCA
  in `X` and `obsm["X_pca"]`, ATAC PCA in `obsm["X_atac"]`, and time, cell type,
  barcode, and stage annotations in `obs`.
- `scmultisim_film_T.pt`: the final time-conditioned FiLM map trained on all paired
  cells and loadable with `coati.maps.load_map`.

Running the tutorial replaces both `scmultisim.h5ad` and `scmultisim_film_T.pt` at
these fixed paths. The simulation parameters are recorded in
[`notebooks/scMultisim/generate_data.R`](../../../notebooks/scMultisim/generate_data.R).

The balanced COATI tutorial uses the main-figure representations and frozen map:

- `all_time_scRNA_pca10.npz`: RNA PCA10 snapshots used for trajectory training.
- `all_time_scATAC_pca8_no_pc2.npz`: paired ATAC PCA8 snapshots.
- `primal_norm_params_rna10_w2.pt` and
  `secondary_norm_params_atac8_no_pc2_w2.pt`: saved W2 normalization scales.
- `scmultisim_rna10_atac8_film_T.pt`: the frozen 10-to-8 FiLM checkpoint used during
  synchronized vector-field training.
- `scmultisim_rna10_umap_model.joblib`: the shared UMAP fitted on all 10,000 observed
  RNA PCA10 cells for the main comparison.
- `observed_rna10_umap.npz`: the matching observed-cell coordinates and population
  labels in that fixed reference UMAP.
