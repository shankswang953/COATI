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

Running the tutorial writes `scmultisim.h5ad` to this fixed path and replaces the
existing file. The simulation parameters are recorded in
[`notebooks/scMultisim/generate_data.R`](../../../notebooks/scMultisim/generate_data.R).
