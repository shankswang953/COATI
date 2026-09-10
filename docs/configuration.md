# Configuration and extension

`Config` retains the original parameter names. Save and reuse settings with `config.save("config.json")` and `Config.load("config.json")`. Unknown top-level keywords raise an error; additional engine fields go in `extra={...}`.

| Parameter | Public default | Purpose |
| --- | --- | --- |
| `niters` | 200 | Small example training budget |
| `num_samples` | 64 | Cells sampled per snapshot; must fit the smallest snapshot |
| `hidden_dim`, `n_layers` | 64, 2 | Original MLP architecture parameters |
| `lr` | 0.001 | AdamW initial learning rate |
| `dt` | 0.1 | Euler training grid; must divide every interval |
| `pdf_coefficient` | 50 | Fixed Sinkhorn penalty |
| `energy_coefficient` | 1 | Primary kinetic-energy coefficient |
| `adaptive_lambda` | False | Opt-in original dual-ascent updates |
| `blur_coeff` | 0.05 | Per-time, per-modality median-distance multiplier |
| `sync_loss` | False | Enable cross-modal constraints |
| `sync_weight` | 0 | Secondary kinetic-energy weight |
| `map_type` | kernel | Scalar kernel; `accurate_kernel` uses per-cell bandwidths |
| `device` | cpu | Explicit public device choice |

Use `config.to_dict()` to inspect every default. These are example defaults, not universal scientific recommendations. Explicitly port your experiment's settings to reproduce that experiment.

## Balanced and unbalanced models

```python
balanced = ct.Config()
synchronized = ct.Config(sync_loss=True, sync_weight=.35)
unbalanced = ct.Config(unbalancedModel=True, alpha_growth=1., mass_loss=True, mass_coefficient=1.)
```

The unbalanced mass prior defaults to cumulative observed counts relative to the first snapshot, as in the original implementation. Set `extra={"biological_num": [...]}` for an explicit prior. The `evaluate()` endpoint metric is unweighted geometric Sinkhorn; use predicted mass and dedicated mass-aware analysis separately for unbalanced models.

## Frozen external map

```python
result = ct.fit(data, ct.Config(sync_loss=True, sync_weight=.35, map_type="MLP"), T_model=my_model)
```

The original `forward(q, t)` contract is retained. COATI copies, moves, freezes and sets the supplied module to evaluation mode; the caller's object remains unchanged. Legacy `map_model` and primary/secondary metric arguments are forwarded. Custom architectures and maps are not serialized automatically; supply them again when needed after reload.

## Normalization

```python
result = ct.fit(data, config, primal_norm_path="rna_scale.pt", sec_norm_path="atac_scale.pt")
```

These are original normalization dictionaries containing `scale`, applied as `x / scale`. `result.predict()` returns input units by default; `raw=False` returns training coordinates. Energy always remains in training units. No normalization is computed by default.

## Time spacing

Training uses the original Euler integrator and energy formulas. The wrapper rejects grids whose intervals are not integer multiples of `dt`, rather than changing their integration silently. Prediction defaults to RK4 on 41 grid points, independently of the training grid. Specify `method="euler"` and an explicit matching grid for a training-discretization comparison.
