# Notebook reproductions

The `examples` directory contains dataset-specific code and fixed assets used to
reproduce the notebooks. These helpers are intentionally kept outside the installed
`coati` package and are not exposed as `ct.*` functions.

| Folder | Purpose |
| --- | --- |
| [`gastrulation/`](gastrulation/) | Mouse gastrulation training, pretrained models, UMAP, map quality and lineage analyses |
| [`palate/`](palate/) | Mouse secondary-palate full-data and strict leave-one-stage-out training |
| [`moscot/`](moscot/) | Mouse pancreas full-data and strict leave-one-stage-out training |
| [`synthetic/`](synthetic/) | Toy data, Gaussian-bump experiments, growth and velocity examples |
| [`data/scmultisim/`](data/scmultisim/) | Prepared scMultiSim assets used by its notebooks |

Run the examples from the repository root with Python's module syntax:

```bash
python -m examples.synthetic.run_toy --dataset gaussian --niters 200 --output outputs/gaussian
python -m examples.gastrulation.train_film --output outputs/gastrulation-film/T_FiLM.pt
python -m examples.gastrulation.train_balanced --map-checkpoint outputs/gastrulation-film/T_FiLM.pt --output outputs/gastrulation-balanced
python -m examples.gastrulation.train_unbalanced --map-checkpoint outputs/gastrulation-film/T_FiLM.pt --output outputs/gastrulation-unbalanced
python -m examples.palate.train_balanced --output outputs/palate-balanced
python -m examples.palate.train_balanced_loso --held-out-stage E13.5 --output outputs/palate-balanced-loo-e13.5
python -m examples.moscot.train_balanced --output outputs/moscot-balanced
python -m examples.moscot.train_unbalanced_loso --output outputs/moscot-unbalanced-loo-e15.5
```

The installed package remains under `src/coati/`. General functionality belongs there;
dataset preparation and notebook reproduction remain here.
