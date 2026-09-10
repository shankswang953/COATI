# COATI

[![COATI overview: inputs, coupled dynamics and model architecture](_static/coati-overview.png)](_static/coati-overview.pdf)

[View overview PDF](_static/coati-overview.pdf)

Learn continuous trajectories from time-indexed snapshots, with optional constraints from a paired second modality. COATI wraps the TraInf research engine in an installable package with explicit inputs, reusable configuration and reloadable results.

```python
import coati as ct

data = ct.datasets.paired()
result = ct.fit(data, ct.Config(sync_loss=True, sync_weight=0.35))
result.plot()
```

Start with a small CPU example, inspect the trajectory, and then substitute your own prepared embeddings. The current validation covers synthetic problems; biological runs are intentionally outside this release's verification.

```{toctree}
:maxdepth: 2
:caption: Get started

installation
quickstart
inputs
tutorials
```

```{toctree}
:maxdepth: 2
:caption: Understand and extend

configuration
api
migration
validation
```
