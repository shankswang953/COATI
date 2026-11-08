# Gastrulation reproductions

These files reproduce the notebooks in `notebooks/gastrulation/`. They are
dataset-specific workflows, not functions in the public `coati` API.

- `train.py`: prepare paired RNA/ATAC inputs and train COATI.
- `dynamics.py`: reproduce the UMAP flow and growth analysis.
- `map_quality.py`: train or reload the RNA-to-ATAC map and inspect its quality.
- `lineage_features.py`: reproduce lineage gene and peak dynamics.
- `pretrained.py`: load the archived full-data checkpoints and original RNA UMAP.
- `config.json`: settings used by the training tutorial.
- `data/pretrained/`: archived checkpoints, RNA50 snapshots and UMAP reducer.

Run a workflow from the repository root, for example:

```bash
python -m examples.gastrulation.train --data-dir /path/to/gastrulation
```
