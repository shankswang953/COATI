# Pretrained full-data gastrulation models

This directory contains two archived 20,000-step full-data checkpoints and the exact
RNA50 snapshots used to train them. They are used by
[`notebooks/gastrulation/pretrainedModels.ipynb`](../../../../notebooks/gastrulation/pretrainedModels.ipynb).

The models are not the strict leave-one-stage-out checkpoints used for the held-out
benchmark. Both use `Cy=0.3`, but they are not a one-parameter ablation:

| Model | Seed | Energy | Density | PDF | Width | Layers | Growth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Balanced | 1 | 0.1 | 0.1 | 100 | 400 | 4 | No |
| Unbalanced | 0 | 0.01 | 0.01 | 100 | 800 | 3 | Yes |

`manifest.json` records the model specifications, training iteration, normalization scale,
biological mass prior and SHA-256 hashes. The compatibility loader reconstructs the exact
architecture and returns a COATI `Result` without modifying or refitting the network. The
The notebook fits the global RNA UMAP directly from all 35,503 rows using 30 neighbors,
`min_dist=0.5`, Euclidean distance, spectral initialization and random seed 42. Keeping
the parameters instead of a serialized reducer avoids a large version-sensitive asset.
