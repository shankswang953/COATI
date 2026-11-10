# Compact palate assets

This directory contains the prepared inputs used by the palate examples. It
does not contain raw RNA counts or ATAC peaks.

- `palate_embeddings.h5ad`: 19,833 paired cells with RNA PCA40, ATAC LSI15,
  supplied RNA/ATAC UMAP coordinates, stage, time, cell type and barcode.
- `primal_norm_params.pt`: fixed RNA normalization scale.
- `secondary_norm_params_lsi15.pt`: fixed ATAC normalization scale.
- `T_FiLM_lsi15.pt`: full-data RNA-to-ATAC map.
- `T_FiLM_lsi15_LOTO_holdout_1.pt`: strict map excluding E13.5.
- `T_FiLM_lsi15_LOTO_holdout_2.pt`: strict map excluding E14.0.

The strict maps must be used for the matching LOSO experiment; the held-out
stage is not used to fit those maps.
