<p align="center"><img src="assets/coati-logo.png" alt="COATI — Cross-Omics Aligned Trajectory Inference" width="400"></p>

<p align="center"><a href="assets/coati-overview.pdf"><img src="assets/coati-overview.png" alt="COATI overview: inputs, coupled dynamics and model architecture" width="100%"></a></p>

[View overview PDF](assets/coati-overview.pdf) · [Watch the animated introduction](https://coati-trajectory-docs.shankswang.chatgpt.site/) · [Manim source](animations/)

# COATI

Cross-Omics Aligned Trajectory Inference. A portable Python interface around the TraInf neural dynamic optimal transport engine, with optional synchronized secondary-modality constraints.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks,docs,dev,visualization]'
```

Run these commands from this repository. The distribution name is `coati-trajectory`; import it as `coati`. This package is not yet published on PyPI.

## Prepare inputs first

COATI recommends Scanpy-compatible fields `celltype`, `time`, `barcode` in
AnnData.obs, matching unique `obs_names`, and `X_umap` in AnnData.obsm.
Common aliases are mapped on a copy; ambiguous columns need an explicit key.

```python
import coati as ct
cells = ct.prepare_anndata(adata, use_rep="X_pca", time_key="day", celltype_key="cell_type")
# Arrays are also assembled automatically, with explicit metadata:
cells = ct.prepare_anndata(embedding, time=cell_times, celltype=cell_types,
                           barcode=cell_ids, compute_umap=True)
result = ct.train_vector_field(cells, balanced=False)
result.plot_dynamics(cells)
```

Unbalanced results show cell-type-colored flow alongside instantaneous growth
with the same streams and labels. Balanced results show motion only. Use
`result.plot_growth`, `result.compute_growth` and `result.write_dynamics(copy=True)`
for individual analyses. Preparation does not normalize features or infer cell
types; reuse prepared inputs and existing UMAPs whenever available.

## Train a vector field

```python
import coati as ct

# Single-modality, balanced training is the default.
data = ct.datasets.gaussian()
result = ct.train_vector_field(data, iterations=200)
trajectory = result.predict_trajectory()
velocity = result.compute_velocity()
result.plot_trajectory()
```

The two basic controls are `multimodal` and `balanced`. A paired run can use:

```python
config = ct.Config(multimodal=True, balanced=False,
                   iterations=500, learning_rate=1e-3, batch_size=64,
                   secondary_weight=0.35)
result = ct.train_vector_field(paired_data, config)
```

Explicit unbalanced mode supplies growth and count-based mass-matching defaults.
You can override every numerical setting. Existing `ct.fit`, `result.predict`,
`result.plot` and original engine parameter names remain supported.

## Velocity on an existing UMAP

```python
# adata uses the same X_pca features/preprocessing as the trained model.
ax = result.plot_velocity(adata, use_rep="X_pca", time_key="time",
                          color="cell_type", style="stream")
ax.figure.savefig("velocity_umap.pdf", bbox_inches="tight")

# Optional: explicitly store vectors in AnnData for downstream use.
result.write_velocity(adata, use_rep="X_pca", time_key="time")
# adata.obsm["coati_velocity"], adata.obsm["coati_velocity_umap"]
```

Use `style="arrows"`, `"grid"` or `"stream"`. Plotting reads the existing UMAP,
keeps AnnData row order and does not modify it. High-dimensional velocities are
projected by a local linear approximation; these are COATI model velocities in
an RNA-velocity-style visualization. See [velocity plotting](docs/velocity.md).

## Scripts and notebooks

```bash
python examples/run_gaussian_sync.py --output outputs/gaussian-sync --iterations 3000
python examples/run_toy.py --dataset split --output outputs/split
python examples/run_toy.py --dataset paired --output outputs/paired
coati toy --dataset split --unbalanced --output outputs/unbalanced
python examples/run_velocity.py --multimodal --unbalanced --output outputs/velocity
python examples/run_dynamics.py --output outputs/dynamics
python examples/run_gastrulation_dynamics.py --data-dir /path/to/gastrulation --output outputs/gastrulation-dynamics --iterations 1000
python examples/run_map_quality.py --data-dir /path/to/gastrulation --output outputs/map-quality
python -m jupyterlab notebooks/
```

Each run directory must be new. Results include the resolved config, input snapshots, checkpoint, TensorBoard logs and metadata. CLI runs also write metrics and a trajectory figure.

- [01 · Gaussian toy: sync vs no sync](notebooks/01_gaussian.ipynb)
- [03 · Splitting and growth](notebooks/03_split.ipynb)
- [04 · Input formats](notebooks/04_inputs.ipynb)
- [06 · Held-out biological distribution reconstruction](notebooks/06_heldout_biological.ipynb)
- [07 · Gastrulation cell types, growth and UMAP flow](notebooks/07_dynamics.ipynb)
- [08 · Train T and inspect biological map quality](notebooks/08_map_quality.ipynb)
- [09 · Lineage gene and peak dynamics](notebooks/09_lineage_features.ipynb)

## Molecular readouts along a lineage

`coati.downstream` supports a gene-name query with `find_gene_peaks` and
`compute_gene_peak_dynamics`, RNA/peak overlays on one axis after separate
temporal z-score normalization, separate plots and optional observed
snapshot overlays. It also provides `select_lineage`, `fit_feature_decoder`,
`compute_feature_dynamics`, curve/heat-map plots and block-based
`read_h5ad_features`. The [lineage tutorial](notebooks/09_lineage_features.ipynb)
checks NMP endpoints and held-out gene/peak readouts before interpretation.
`start_time` on `predict_trajectory` supports a supplied later-stage initial cohort.
Training objectives and the original TraInf repositories are unchanged.

Run `examples/run_lineage_features.py --help` for the script counterpart. Molecular
features must be supplied separately from PCA/LSI, in verified barcode order.
Local feature interpolation is an estimate, not an inverse embedding or evidence
of regulation. The current 1,000-step example has no accepted NMP → Somitic cohort;
its additional curves are labeled NMP-origin diagnostics.

## Your own data

```python
data = ct.TemporalData.from_npz(
    "primary.npz", secondary_path="secondary.npz",
    keys=["early", "middle", "late"], times=[0.0, 0.5, 1.0],
)
# Also: TemporalData(list_of_arrays, times), from_dict, from_csv,
# from_dataframe, from_anndata and from_h5ad.
```

Cross-modal inputs must be paired **within each snapshot**, including row identity and ordering. Across-time cells may be unpaired and counts may differ. Inputs are used as supplied: no hidden normalization, PCA or log transform.

## External-method benchmarks

[TraInfBench](https://github.com/shankswang953/TraInfBench) hosts the external-method portion of our benchmark comparison, including the code for running external methods and comparing them with COATI.

## Documentation

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

The documentation uses Sphinx + Furo and includes a Read the Docs configuration. See [migration](docs/migration.md) for the exact scope of the packaging changes and [validation](docs/validation.md) for verification results.

## Structure

```text
src/coati/           public data, config, training and result interfaces
src/coati/maps.py    automatic FiLM/MLP fitting, validation and checkpoints
src/coati/_core/     preserved research engine with package-relative imports
examples/           script entry point
notebooks/          executable tutorials
tests/              data and training checks
scripts/            original-engine equivalence verification
docs/               Sphinx documentation source
provenance/         source hashes and validation records
assets/             supplied COATI logo
```

The original TraInf repositories are not dependencies and are not modified. Defaults are explicit toy-oriented choices; the biological holdout notebook is a short reconstruction example, not a claim of convergence or biological superiority. Reference interface inspiration: [CytoBridge](https://github.com/zhenyiizhang/CytoBridge) and its [documentation](https://cytobridge.readthedocs.io/en/latest/index.html). No CytoBridge implementation is copied.
