# Gastrulation reproductions

This folder contains the dataset-specific mouse gastrulation workflow used by
the notebooks. It is intentionally outside the public `coati` API.

## Included data

`data/gastrulation_embeddings.h5ad` is the lightweight paired dataset used by
TraInf: 35,503 cells with RNA PCA50 in `obsm['X_pca']`, ATAC LSI14 in
`obsm['X_atac']`, the original RNA UMAP, cell type, stage, time and barcode.
It does not contain the multi-gigabyte RNA or ATAC count matrices.

The folder also contains the exact TraInf normalization dictionaries and the
archived FiLM checkpoint:

- `primal_norm_params.pt`: RNA scale `22.863716719648387`.
- `secondary_norm_params.pt`: ATAC scale `0.0015835087817000948`.
- `T_FiLM.pt`: the original 50-to-14 FiLM map selected at epoch 150.

`build_embeddings.py` records how the compact AnnData was assembled from the
original TraInf NPZ files. `gastrulation_embeddings.json` records source and
output checksums.

## Exact TraInf order of operations

1. Split the AnnData into E7.5, E8.0, E8.5 and E8.75 snapshots.
2. Divide RNA PCA and ATAC LSI independently by their saved global scales.
3. Train the original FiLMMLP for 300 epochs using paired MSE.
4. Select T by the summed held-out Sinkhorn score across all four time points.
5. Freeze T and start balanced or unbalanced vector-field training.

The dataset-specific implementation is in `film.py`; it does not use COATI's
newer standardized `PairedMap`, because that would not match TraInf.

From the repository root:

```bash
source .venv/bin/activate

python -m examples.gastrulation.train_film \
  --output outputs/gastrulation-film/T_FiLM.pt

python -m examples.gastrulation.train_balanced \
  --map-checkpoint outputs/gastrulation-film/T_FiLM.pt \
  --output outputs/gastrulation-balanced

python -m examples.gastrulation.train_unbalanced \
  --map-checkpoint outputs/gastrulation-film/T_FiLM.pt \
  --output outputs/gastrulation-unbalanced
```

The bundled `data/T_FiLM.pt` is the default if `--map-checkpoint` is omitted.
For a smoke test, use `--epochs 5` when fitting T and `--iterations 20` when
fitting either vector field.

Both full vector-field configurations run for 20,000 iterations and require
`device="mps"`. They fail immediately when Apple MPS is unavailable rather
than silently falling back to CPU. Check the first line of `training.log`
before leaving a long run; it must report `device = mps`.

For a detached manual run on one MPS device, run the two models sequentially
inside one `tmux` session:

```bash
tmux new-session -d -s coati-gastrulation \
  'cd /Users/jingfeng/local/project/COATI && \
  .venv/bin/python -m examples.gastrulation.train_balanced \
    --map-checkpoint outputs/gastrulation-film/T_FiLM.pt \
    --output outputs/gastrulation-balanced-mps && \
  .venv/bin/python -m examples.gastrulation.train_unbalanced \
    --map-checkpoint outputs/gastrulation-film/T_FiLM.pt \
    --output outputs/gastrulation-unbalanced-mps'
```

`balanced.json` uses the architecture and loss settings from
`TraInf/Gastrulation/FilmSync`, with the requested 20,000-iteration run length.
`unbalanced.json` mirrors `TraInf/Gastrulation/UnbalancedSync_biological_num`.
These current FiLM recipes are distinct from the older archived checkpoints
shown in `pretrainedModels.ipynb`; the notebook states that provenance
explicitly rather than presenting the old weights as FiLM reproductions.

Other scripts in this folder reproduce UMAP dynamics, map-quality and lineage
feature analyses.
