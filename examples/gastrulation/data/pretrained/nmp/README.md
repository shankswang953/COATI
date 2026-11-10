# NMP pretrained models

These are the final checkpoints from the current 20,000-step full-data mouse
gastrulation runs used in `notebooks/gastrulation/nmpTrajectoryComposition.ipynb`.
Both use seed 0, RNA PCA50, ATAC LSI14, the committed FiLM map, and `C_y=0.5`.

Only the vector-field state and iteration are stored. Optimizer state,
TensorBoard events, logs, and intermediate checkpoints are intentionally
excluded. Exact file hashes and model configurations are recorded in
`manifest.json`.
