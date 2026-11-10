# Compact gastrulation assets

This folder contains only prepared embeddings and small model assets. It does
not contain raw RNA counts, ATAC peaks or the multi-gigabyte source AnnData.

- `gastrulation_embeddings.h5ad`: 35,503 paired cells; RNA PCA50, ATAC LSI14,
  RNA UMAP and observation metadata.
- `primal_norm_params.pt`: RNA normalization scale used by COATI.
- `secondary_norm_params.pt`: ATAC normalization scale used by COATI.
- `T_FiLM.pt`: archived COATI FiLM checkpoint.
- `gastrulation_embeddings.json`: shapes, time-point counts and SHA-256
  provenance.
- `pretrained/`: older vector-field checkpoints used by the pretrained-model
  inference notebook.

The top-level `build_embeddings.py` reconstructs the compact AnnData from the
prepared PCA/LSI NPZ files and verifies row identity against the
processed RNA AnnData before writing anything.
