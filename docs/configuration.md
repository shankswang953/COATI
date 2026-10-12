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
| `map_type` | film | Automatically trained FiLM; `mlp` omits time; `kernel` and `accurate_kernel` retain explicit kernel modes |
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

The supplied map may implement `forward(q, t)` or `forward(q)`; COATI adapts the latter to ignore time. It must support inputs with arbitrary leading dimensions, including a time-by-particle tensor. COATI copies, moves, freezes and sets the supplied module to evaluation mode; the caller's object remains unchanged. It does not retrain the supplied map. Legacy `map_model` and primary/secondary metric arguments are forwarded. Custom architectures and maps are not serialized automatically; supply the same trained map again after reload. Providing a custom map does not establish which stages it was trained on; that exclusion remains the caller's responsibility.

Without an external map, synchronized runs train and save a FiLM map automatically. `map_type="mlp"` selects a time-independent MLP. Pass `map_config=ct.MapConfig(...)` to configure epochs, validation splits and quality warnings. These settings apply only to automatically trained maps. See [Cross-modal maps](maps.md).

## Normalization

```python
result = ct.fit(data, config, primal_norm_path="rna_scale.pt", sec_norm_path="atac_scale.pt")
```

These are original normalization dictionaries containing `scale`, applied as `x / scale`. `result.predict()` returns input units by default; `raw=False` returns training coordinates. Energy always remains in training units. No normalization is computed by default.

## Time spacing

Training uses the original Euler integrator and energy formulas. The wrapper rejects grids whose intervals are not integer multiples of `dt`, rather than changing their integration silently. Prediction defaults to RK4 on 41 grid points, independently of the training grid. Specify `method="euler"` and an explicit matching grid for a training-discretization comparison.

### Distribution matching

Matching uses the existing W2-based Sinkhorn loss (`p=2`). MMD matching and custom
Riemannian metric arguments are no longer supported. This does not remove
`result.evaluate()` or its evaluation metrics. See [migration](migration.md)
for loading older configurations.

## Clear mode controls and parameter names

`Config(multimodal=False, balanced=True)` selects the basic single-modality balanced
model. Explicit multimodal=True supplies secondary_weight=.35; explicit balanced=False
supplies growth_weight=1, mass_loss=True and mass_weight=1. Overrides remain available.
Calling Config() or using original engine names retains the original defaults.

| Clear name | Existing engine name |
| --- | --- |
| iterations | niters |
| batch_size | num_samples |
| learning_rate | lr |
| secondary_weight | sync_weight |
| matching_weight | pdf_coefficient |
| manifold_weight | density_coefficient |
| energy_weight | energy_coefficient |
| growth_weight | alpha_growth |
| mass_weight | mass_coefficient |

Supplying both names for the same parameter is rejected. `config.to_dict()` and saved
JSON retain engine keys for reproducibility. `config.with_parameters(...)` returns a
new configuration. `train_vector_field(data, config, iterations=500)` uses the same
mechanism, leaving the supplied configuration unchanged. Explicit mode changes select
the preset for the new mode; explicitly supplied numerical overrides take precedence.
