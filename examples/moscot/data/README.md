# Compact pancreas assets

This directory contains the prepared inputs used by the pancreas examples. It
does not contain raw RNA counts or ATAC peaks.

- `moscot_embeddings.h5ad`: 22,604 paired cells with RNA PCA50, ATAC
  PoissonVI22, supplied RNA/ATAC UMAP coordinates, stage, time, cell type and
  barcode.
- `primal_norm_params.pt`: fixed RNA normalization scale.
- `secondary_norm_params_poissonvi.pt`: fixed ATAC normalization scale.
- `T_FiLM_poissonvi.pt`: full-data RNA-to-ATAC map.
- `T_FiLM_poissonvi_LOTO_holdout_1.pt`: strict map excluding E15.5.

The strict map must be used for LOSO; E15.5 is not used to fit that map.
