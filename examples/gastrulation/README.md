# Gastrulation reproductions

This folder contains the dataset-specific mouse gastrulation workflow used by
the notebooks. It is intentionally outside the public `coati` API.

## Included data

`data/gastrulation_embeddings.h5ad` is the lightweight paired dataset used by
COATI: 35,503 cells with RNA PCA50 in `obsm['X_pca']`, ATAC LSI14 in
`obsm['X_atac']`, the original RNA UMAP, cell type, stage, time and barcode.
It does not contain the multi-gigabyte RNA or ATAC count matrices.

The folder also contains the normalization dictionaries and the
archived FiLM checkpoint:

- `primal_norm_params.pt`: RNA scale `22.863716719648387`.
- `secondary_norm_params.pt`: ATAC scale `0.0015835087817000948`.
- `T_FiLM.pt`: the original 50-to-14 FiLM map selected at epoch 150.

`build_embeddings.py` records how the compact AnnData was assembled from the
prepared PCA/LSI files. `gastrulation_embeddings.json` records source and
output checksums.

## COATI training workflow

1. Split the AnnData into E7.5, E8.0, E8.5 and E8.75 snapshots.
2. Divide RNA PCA and ATAC LSI independently by their saved global scales.
3. Train the dataset-specific FiLMMLP for 300 epochs using paired MSE.
4. Select T by the summed held-out Sinkhorn score across all four time points.
5. Freeze T and start balanced or unbalanced vector-field training.

The dataset-specific implementation is in `film.py`. It is kept separate from
the general `PairedMap` interface so this example reproduces the published
gastrulation configuration exactly.

## Strict leave-one-stage-out training

The normalization parameters are fixed from all four observed stages. The held-out
stage is then removed from both FiLM-map fitting and trajectory fitting; the supplied
LOTO FiLM checkpoints already follow that split.

```bash
python -m examples.gastrulation.train_balanced_loso \
  --held-out-stage E8.0 \
  --output outputs/gastrulation-balanced-loo-e8.0

python -m examples.gastrulation.train_unbalanced_loso \
  --held-out-stage E8.0 \
  --output outputs/gastrulation-unbalanced-loo-e8.0
```

Use `--held-out-stage E8.5` for the other middle stage. The unbalanced mass prior
is reduced from `1:4:6:11` to the retained stages automatically.

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

Both full vector-field configurations run for 20,000 iterations. The public
configuration uses `device="auto"`: COATI selects CUDA when available, then
Apple MPS, and otherwise CPU. Pass `--device cuda`, `--device mps` or
`--device cpu` to override this selection explicitly. Check the first line of
`training.log` before leaving a long run to confirm the selected device.

The RNA-only controls use the same RNA PCA50 snapshots and RNA normalization
but omit the ATAC snapshots and FiLM map entirely. Their entry points are
`train_balanced_rna_only` and `train_unbalanced_rna_only`. In the unbalanced
configuration, the transport coefficient is implicitly one and
`growth_weight=1.0` sets the continuity-equation growth coefficient to one;
`mass_weight=1.0` is the separate mass-matching penalty.

For a detached manual run, run the two models sequentially inside one `tmux`
session from the repository root:

```bash
tmux new-session -d -s coati-gastrulation \
  'python -m examples.gastrulation.train_balanced \
    --map-checkpoint outputs/gastrulation-film/T_FiLM.pt \
    --output outputs/gastrulation-balanced && \
  python -m examples.gastrulation.train_unbalanced \
    --map-checkpoint outputs/gastrulation-film/T_FiLM.pt \
    --output outputs/gastrulation-unbalanced'
```

`balanced.json` and `unbalanced.json` record the architecture and loss settings
used for the corresponding 20,000-iteration COATI runs.
These current FiLM recipes are distinct from the older archived checkpoints
shown in `pretrainedModels.ipynb`; the notebook states that provenance
explicitly rather than presenting the old weights as FiLM reproductions.

Other scripts in this folder reproduce UMAP dynamics, map-quality and lineage
feature analyses.

## Reproduce the NMP-related UMAPs

`nmp_umap.py` fits the RNA and ATAC reference UMAPs directly from
`data/gastrulation_embeddings.h5ad`. It does not load or save a serialized UMAP
reducer. The script keeps the original AnnData row order and selects the seven
NMP-related cell types before fitting.

Exact coordinate reproduction requires the visualization extra, which pins
`umap-learn==0.5.9.post2`:

```bash
python -m pip install -e '.[visualization]'
python -m examples.gastrulation.nmp_umap
```

RNA uses 100 neighbors, `min_dist=0.8`, Euclidean distance and seed 42. ATAC
uses 150 neighbors, `min_dist=0.98`, Euclidean distance, seed 0 and
`angular_rp_forest=True`. Both use spectral initialization, `transform_seed=42`
and one worker. The output contains coordinates, figures and provenance, but no
reducer object.

## Held-out distribution evaluation

After the four E8.0/E8.5 LOSO runs finish, compare each predicted held-out
distribution with the observed cells in normalized RNA PCA50 and ATAC LSI14:

```bash
python -m examples.gastrulation.evaluate_loso
```

The command writes `outputs/gastrulation-loso-metrics/six_metrics.csv`. It
reports support recall, Sinkhorn-based W2, and cell-type composition similarity
independently in RNA and ATAC. Balanced runs use uniform particle weights;
unbalanced runs use their normalized predicted masses. The metrics compare
distribution shape and composition, not total population size.
