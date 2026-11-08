# Notebook reproductions

The `examples` directory contains dataset-specific code and fixed assets used to
reproduce the notebooks. These helpers are intentionally kept outside the installed
`coati` package and are not exposed as `ct.*` functions.

| Folder | Purpose |
| --- | --- |
| [`gastrulation/`](gastrulation/) | Mouse gastrulation training, pretrained models, UMAP, map quality and lineage analyses |
| [`synthetic/`](synthetic/) | Toy data, Gaussian-bump experiments, growth and velocity examples |
| [`data/scmultisim/`](data/scmultisim/) | Prepared scMultiSim assets used by its notebooks |

Run the examples from the repository root with Python's module syntax:

```bash
python -m examples.synthetic.run_toy --dataset gaussian --niters 200 --output outputs/gaussian
python -m examples.gastrulation.train --data-dir /path/to/gastrulation
```

The installed package remains under `src/coati/`. General functionality belongs there;
dataset preparation and notebook reproduction remain here.
