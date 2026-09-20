# Quick start

## A complete Python run

```python
import coati as ct

data = ct.datasets.gaussian(n=128, seed=0)
config = ct.Config(niters=200, num_samples=64, seed=0)
result = ct.fit(data, config)
trajectory = result.predict()
print(trajectory["primary"].shape)  # time × cells × features
print(result.evaluate())
ax = result.plot()
```

`fit` returns a `Result` and saves a unique run under `outputs/`. Use `output_dir="outputs/my-run"` for a named, empty directory. Reusing a nonempty directory raises an error, protecting previous notebook results.

## Reload without retraining

```python
restored = ct.load_result(result.output_dir)
restored.predict()
```

## Paired modalities with an automatically trained map

```python
data = ct.datasets.paired(n=128)
result = ct.fit(data, ct.Config(sync_loss=True, sync_weight=.35))
print(result.map_report["status"])
secondary = result.project_secondary()
restored = ct.load_result(result.output_dir)  # restores the FiLM map too
```

The map is fitted to paired cells, validated within each supplied stage and frozen before trajectory fitting. Choose `map_type="mlp"` for a map without time input. The run contains `map/report.txt`, `map/report.json`, training history and the saved map. Quality warnings are visible even when trajectory logs are redirected. See [Cross-modal maps](maps.md) for held-out stages, custom maps and the configurable quality gate.

## Command line

```bash
python examples/run_toy.py --dataset gaussian --output outputs/gaussian
coati toy --dataset paired --niters 200 --output outputs/paired
```

Each CLI run writes `metrics.json` and `trajectories.png`, as well as model and training artifacts. Training diagnostics go to `training.log`; use `fit(..., verbose=True)` to print them interactively.

## Interpreting a toy run

Compare endpoint Sinkhorn before and after training against the measured sampling floor. Short training is a functional sanity check, not proof of convergence. Kinetic energy is in training coordinates; the familiar normalized lower bound near one applies only when the corresponding global W2 normalization was actually applied.
