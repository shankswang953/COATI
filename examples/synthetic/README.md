# Synthetic reproductions

These files reproduce the synthetic notebooks. They contain toy-data generation,
Gaussian-bump helpers, fixed assets and command-line workflows; they are not part
of the public `coati` API.

- `toy_data.py`: small Gaussian, branching and paired datasets.
- `run_toy.py`, `run_dynamics.py`, `run_velocity.py`: general toy workflows.
- `gaussian_*`: Gaussian-bump generation, mapping, comparison and replay helpers.
- `build_gaussian_assets.py`: regenerate the committed Gaussian AnnData and map.
- `data/gaussian_bump/`: shared assets used by the active synthetic notebooks.
- `data/gaussian_reference/`: unchanged artifacts from the original experiment.

Run a workflow from the repository root, for example:

```bash
python -m examples.synthetic.run_toy --dataset gaussian --niters 200 --output outputs/gaussian
```
