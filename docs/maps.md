# Cross-modal maps

## Train T without a vector field

The existing `coati.maps.train_map` function can fit T independently. The
[biological map tutorial](tutorials/08_map_quality.ipynb) shows paired gastrulation
inputs, map fitting, checkpoint reload, a shared ATAC UMAP and cell-type/stage
checks on unseen pairs.

```python
import coati as ct
from coati.maps import train_map, load_map

paired = ct.TemporalData.from_anndata(cells, secondary_obsm_key="X_atac")
T, report = train_map(paired, kind="film", config=ct.MapConfig(epochs=300),
                      directory="outputs/new-map", device="cpu", seed=0)
restored_T = load_map("outputs/new-map/model.pt", device="cpu")
```

`cells` is a prepared AnnData with the selected primary representation and row-paired `X_atac` features. Use a fresh directory for each map. `T` accepts primary training features and
numeric times, and returns secondary features in the corresponding training
units. `kind="mlp"` selects a map without time conditioning. The returned model
is frozen and can subsequently be supplied as `T_model=T` to synchronized
trajectory training, with matching input preprocessing and normalization.

Internal validation selects the checkpoint. For independent paired-cell checks,
reserve test RNA/ATAC rows together before calling `train_map`. Fit any reference
classifier and ATAC UMAP on optimization rows, then use that same UMAP's
`transform` on both true and mapped test ATAC. UMAP resemblance is a visual check;
quantify label agreement in the original secondary features. For FiLM, stage
consistency is conditional on the provided time, not independent time inference.

The same biological example runs as a script:

```bash
python examples/run_map_quality.py --data-dir /path/to/gastrulation --output outputs/map-quality
```

Use `--map-kind mlp` for time-independent fitting or `--reuse` with an existing
run to repeat its quality checks without retraining.

## Default paired training

```python
import coati as ct

data = ct.datasets.paired(n=128)
result = ct.fit(data, ct.Config(sync_loss=True, sync_weight=.35))
print(result.map_report)
```

With synchronization enabled and no supplied map, `fit` first trains a FiLM network on the paired observations. Time conditions the hidden layers through feature-wise affine modulation. The map is then frozen, while gradients through its inputs continue to constrain the RNA trajectory. No external map-training script or checkpoint is needed.

`Config(map_type="mlp", sync_loss=True, sync_weight=.35)` selects the same main MLP architecture without time conditioning. `kernel` and `accurate_kernel` remain available and do not fit a neural map. Single-modality runs do not train a map.

## Training settings

```python
map_config = ct.MapConfig(epochs=300, batch_size=1024, patience=30,
                          validation_fraction=.2, quality_action="warn")
result = ct.fit(data, ct.Config(sync_loss=True, sync_weight=.35),
                map_config=map_config, output_dir="outputs/my-paired-run")
```

The default network has two 128-unit GELU hidden layers and a linear output layer. FiLM uses a 32-unit time encoder whose modulation output is initialized to zero. Default optimization uses AdamW with learning rate `1e-3` and weight decay `1e-4`, for at most 300 epochs. Training minimizes paired MSE after standardizing target features. Input and target means and standard deviations are estimated only from the training rows; the network converts its output back to the trajectory coordinate scale. FiLM time is rescaled using the minimum and maximum supplied times. MLP does not use time.

Within each supplied stage, 20% of paired rows are reserved for validation, with at least two training and two validation cells. RNA and ATAC row identities are kept together. The best checkpoint by pooled standardized validation MSE is retained, with early stopping after 30 epochs without improvement. The trajectory seed also controls map initialization and the split. Validation rows are used for map checkpoint selection, so these are validation diagnostics, not independent test metrics. All supplied rows remain available for subsequent trajectory fitting.

`primal_norm_path` and `sec_norm_path`, when supplied to `fit`, are applied before fitting the map, consistently with trajectory training. Map input/output standardization is internal and is saved in the checkpoint; it does not alter the kinetic cost or the data stored in `TemporalData`.

## Held-out stages and prediction times

```python
observed = ct.datasets.paired(n=128)  # times 0, 0.5, 1
training = observed.without_times([0.5])
result = ct.fit(training, ct.Config(sync_loss=True, sync_weight=.35))
prediction = result.predict(times=[0., .5, 1.])
secondary = result.project_secondary(prediction)
assert result.map_report["training_times"] == [0., 1.]
```

Alternatively, construct `TemporalData` or select NPZ keys using only the retained stages. There is no full-atlas fallback: the automatic map, its statistics, validation splits and trajectory fitting all use only the supplied snapshots. Times are not renumbered after exclusion. A missing intermediate stage is simply a prediction query time. At least two observed stages must remain, and `dt` must divide their time intervals.

This excludes stages from the fitted map and trajectory fields, not from preprocessing done outside COATI. Within-stage validation cannot establish accuracy at unseen times. When supplying an already trained map, the caller is responsible for excluding the held-out stage from its training.

## Quality reports

Each automatically trained map writes `map/report.txt` and `map/report.json`. `result.map_report` exposes the JSON report. Per-stage diagnostics include

- Paired RMSE and MSE in trajectory-training coordinates, calculated on all validation pairs.
- Normalized paired MSE, divided by the MSE of predicting the corresponding training-stage ATAC mean. Lower is better; one means no improvement over this baseline.
- Debiased Sinkhorn divergence with half squared Euclidean cost and an empirical observed split reference. Each comparison uses equally sized validation samples, at most 256 cells per sample, over five repeats. The regularization length is 0.05 times a median validation pairwise distance. These sampling choices and lengths are recorded per stage.
- Training/validation counts, supplied times, best epoch and any quality warnings.

By default, a stage raises a warning when normalized paired MSE exceeds 0.5, the Sinkhorn/split-reference ratio exceeds 5, or fewer than 20 validation cells are available. Undefined ratios also produce a warning. These configurable thresholds are screening heuristics, not universal scientific acceptance criteria. Distributional similarity alone cannot establish correct paired or cell-type correspondence, and neither metric validates trajectories at omitted stages.

Warnings are emitted before trajectory fitting and also saved in the reports. `MapConfig(quality_action="error")` saves the map and report but stops before fitting trajectories when any quality warning is present. Non-finite training or validation losses always stop training and write a failure report. Inspect the report and data pairing before increasing the training budget or changing a threshold.

Training curves are stored in `map/history.json`, and `map/splits.npz` stores row indices within each supplied snapshot. `map/config.json` records the map settings. `map/model.pt` includes architecture and normalization buffers, so `ct.load_result(...)` restores the built-in map without retraining.

## User-provided maps

```python
result = ct.fit(data, ct.Config(sync_loss=True, sync_weight=.35), T_model=my_map)
```

A PyTorch module can implement `forward(q, t)` or a time-independent `forward(q)`. It must map normalized RNA training coordinates to normalized ATAC training coordinates and preserve leading dimensions such as time-by-particle. COATI copies and freezes it without changing the caller's module. No map fitting or validation claim is made for this external model; `result.map_report` is `None`. Supply the same trained external map to `ct.load_result(..., T_model=my_map)` to restore secondary predictions.

## Command line

```bash
coati toy --dataset paired --output outputs/film
coati toy --dataset paired --map-type mlp --map-epochs 300 --output outputs/mlp
python examples/run_paired_loso.py --output outputs/paired-loso
```

For `coati fit`, `--map-config map.json` accepts the fields of `MapConfig`. The main configuration selects `map_type` and enables synchronization.
