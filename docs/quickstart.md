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

## Command line

```bash
python examples/run_toy.py --dataset gaussian --output outputs/gaussian
coati toy --dataset paired --niters 200 --output outputs/paired
```

Each CLI run writes `metrics.json` and `trajectories.png`, as well as model and training artifacts. Training diagnostics go to `training.log`; use `fit(..., verbose=True)` to print them interactively.

## Interpreting a toy run

Compare endpoint Sinkhorn before and after training against the measured sampling floor. Short training is a functional sanity check, not proof of convergence. Kinetic energy is in training coordinates; the familiar normalized lower bound near one applies only when the corresponding global W2 normalization was actually applied.
