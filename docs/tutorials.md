# Cell-by-cell tutorials

Start JupyterLab with `python -m jupyterlab notebooks/` after installation. Each notebook has separate cells for imports, data inspection, configuration, training, prediction, metrics, plotting and reload.

| Notebook | What it covers |
| --- | --- |
| [Gaussian translation](_static/notebooks/01_gaussian.html) | Basic balanced transport across three snapshots |
| [Paired modalities](_static/notebooks/02_paired.html) | 2D primary → 3D secondary kernel synchronization |
| [Splitting and growth](_static/notebooks/03_split.html) | Two terminal branches with doubled observed counts |
| [Input adapters](_static/notebooks/04_inputs.html) | Arrays, dictionary, NPZ, CSV and AnnData equivalence |

## A paired run, step by step

```python
import coati as ct
```

```python
data = ct.datasets.paired(n=128, seed=0)
[(x.shape, y.shape) for x, y in zip(data.primary, data.secondary)]
```

```python
config = ct.Config(niters=200, sync_loss=True, sync_weight=.35)
result = ct.fit(data, config)
```

```python
trajectory = result.predict()
result.plot()
result.evaluate()
```

These are runnable functional examples, not claims of superior trajectory accuracy or biological performance.

## Download notebooks

- [Download 01_gaussian.ipynb](_static/notebooks/01_gaussian.ipynb)
- [Download 02_paired.ipynb](_static/notebooks/02_paired.ipynb)
- [Download 03_split.ipynb](_static/notebooks/03_split.ipynb)
- [Download 04_inputs.ipynb](_static/notebooks/04_inputs.ipynb)

## Project secondary trajectories

```python
secondary = result.project_secondary(trajectory)
```

Kernel prediction uses all paired observations as its deterministic basis; training still uses the original randomly sampled paired basis. Both primary and secondary outputs default to input units.
