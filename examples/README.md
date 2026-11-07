# COATI examples

Data preparation and experiment-specific helpers live here, outside the installed
`coati` package. Clone the repository to use these scripts and notebooks.

- `toy_data.py`: small synthetic datasets and an analytic Gaussian-bump map.
- `generate_gaussian_data.py`: the original two-stage Gaussian data recipe.
- `gaussian_parameters.py`: concise COATI configuration for the fresh comparison.
- `gaussian_map.py`: original Fourier map fitting and held-out quality checks.
- `gaussian_fresh.py`: fresh observed/support AnnData assembly.
- `gaussian_sync.json`: retained settings for the earlier analytic-map comparison.
- `gaussian_comparison.py`: comparison setup and plots used by that tutorial.
- `gaussian_reference.py`: loading and replaying the unchanged original artifacts.
- `data/gaussian_reference/`: original NPZ files, checkpoints and reference outputs.

Run a small example from the repository root:

```bash
python examples/run_toy.py --dataset gaussian --niters 200 --output outputs/gaussian
```

The package CLI trains on prepared inputs with `coati train-vector-field` (or
`coati fit`). Synthetic data generation is not a package command.

## Original Gaussian-bump data

The original recipe has two stages:

1. `GaussToy/data/generate_gaussian_toy.py` samples a Gaussian mixture by rejection
   inside `[0, 1]^2`, with NumPy seed `2026`. The initial snapshot has 1000 points
   centered at `(0.30, 0.30)`. The terminal snapshot contains two groups of 1000
   points centered at `(0.56, 0.82)` and `(0.83, 0.56)`.
2. `GaussToy/GaussianBump/generate_bump_data.py` constructs secondary observations
   and separate support pairs for fitting the cross-modal map. The bump has
   center `(0.5, 0.5)`, width `0.08` and amplitude `1`. The saved observed secondary
   snapshots have **z = 0**. Support pairs follow the bump surface instead:
   `z = exp(-((x-0.5)^2 + (y-0.5)^2) / (2 * 0.08^2))`.

The covariance matrices, before truncation, are:

```text
initial:    [[0.0042,  0.0002], [ 0.0002, 0.0038]]
terminal A: [[0.0030,  0.0001], [ 0.0001, 0.0030]]
terminal B: [[0.0032, -0.0001], [-0.0001, 0.0032]]
```

The support has 4000 uniform points in `[0.02, 0.98]^2` and 3000 truncated
central points with standard deviation `0.13`, generated with seed `2026`.

Prepare equivalent arrays in a new directory without invoking the package:

```bash
python examples/generate_gaussian_data.py --output outputs/gaussian-data
```

This writes `rna_2d_by_time.npz`, `secondary_3d_bump_by_time.npz` and
`bump_support_2d_3d.npz`. The default arrays match the original saved arrays.
The script refuses to overwrite nonempty directories.

The original-reference notebook reads the archived files directly. It never
regenerates them; their file hashes are retained in
`data/gaussian_reference/manifest.json`. The active synthetic tutorials share
the committed AnnData and frozen map in `examples/data/synthetic/gaussian_bump/`.
The build script regenerates those assets from the same repository code and refuses
to overwrite an existing asset directory.

The fresh AnnData workflow requires the `anndata` extra; Jupyter additionally
requires the `notebooks` extra.

```bash
python examples/build_synthetic_gaussian_assets.py \
  --output outputs/gaussian-bump-assets
```

```bash
python examples/run_gaussian_sync.py --output outputs/gaussian-fresh
python examples/run_gaussian_sync.py --map-only --output outputs/gaussian-map
```

`--map-steps` and `--iterations` allow short functional checks. The earlier
analytic-map preset in `gaussian_sync.json` is retained separately; the active
fresh-data workflow does not use it.
