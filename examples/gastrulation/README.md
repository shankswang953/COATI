# Gastrulation reproductions

These files reproduce the notebooks in `notebooks/gastrulation/`. They are
dataset-specific workflows, not functions in the public `coati` API.

- `train_balanced.py`: train the archived balanced full-data recipe.
- `train_unbalanced.py`: train the archived unbalanced full-data recipe.
- `train.py`: shared dataset loader and command-line implementation.
- `dynamics.py`: reproduce the UMAP flow and growth analysis.
- `map_quality.py`: train or reload the RNA-to-ATAC map and inspect its quality.
- `lineage_features.py`: reproduce lineage gene and peak dynamics.
- `pretrained.py`: load the archived full-data checkpoints and original RNA UMAP.
- `balanced.json` and `unbalanced.json`: the two full-data training settings.
- `config.json`: shorter settings used by the introductory training tutorial.
- `data/pretrained/`: archived checkpoints, RNA50 snapshots and UMAP reducer.

The input directory must contain paired `rna.h5ad` and `atac.h5ad` files with
identically ordered barcodes. Run either full-data configuration from the repository root:

```bash
python -m examples.gastrulation.train_balanced \
  --data-dir /path/to/gastrulation \
  --output outputs/gastrulation-balanced

python -m examples.gastrulation.train_unbalanced \
  --data-dir /path/to/gastrulation \
  --output outputs/gastrulation-unbalanced
```

For a quick check before a full 20,000-step run, add `--iterations 20`. The two
JSON files record the settings associated with the archived checkpoints. The
historical frozen cross-modal map was not stored with those vector-field
checkpoints, so these commands fit a new map from the supplied paired RNA/ATAC
data and are not expected to reproduce the checkpoint weights byte for byte.
