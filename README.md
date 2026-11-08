<p align="center"><img src="assets/coati-logo.png" alt="COATI — Cross-Omics Aligned Trajectory Inference" width="100%"></p>

[Website](https://shankswang953.github.io/COATI_website/) · [Tutorials](https://shankswang953.github.io/COATI_website/tutorials.html)

# COATI

**Cross-Omics Aligned Trajectory Inference**

COATI learns continuous cell-state trajectories from time-indexed single-cell snapshots, with optional synchronization across RNA and ATAC modalities. It supports single- and multimodal inputs, balanced and unbalanced dynamics, and downstream trajectory analysis.

## Install

Use Python 3.10 or newer:

```bash
python -m pip install "coati-trajectory[notebooks,visualization] @ git+https://github.com/shankswang953/COATI.git"
```

The package is imported as `coati`. For local development and the notebooks:

```bash
git clone https://github.com/shankswang953/COATI.git
cd COATI
python -m pip install -e '.[notebooks,visualization]'
```

## Structure

```text
src/coati/     Python package and training engine
examples/      Data preparation and standalone Python examples
notebooks/     Executable tutorials
tests/         Package tests
assets/        Package logo
```

Toy-data preparation and Gaussian-specific helpers are maintained in [examples](examples/README.md), outside the installed package.

## Notebooks

- [Introduction](notebooks/00_introduction.ipynb)
- [Original Gaussian experiment](notebooks/11_gaussian_reference.ipynb)
- [Synthetic data](notebooks/syntheticdata/)
  - [Balanced Gaussian bump: sync vs no sync](notebooks/syntheticdata/balancedGaussianbump.ipynb)
  - [Unbalanced Gaussian bump: trajectories and growth](notebooks/syntheticdata/unbalancedGaussianbump.ipynb)
  - [Cy comparison: 0.1, 0.5 and 0.7](notebooks/syntheticdata/cyComparison.ipynb)
- [Splitting and growth](notebooks/03_split.ipynb)
- [Input formats](notebooks/04_inputs.ipynb)
- [Gastrulation](notebooks/gastrulation/)
  - [Mouse gastrulation: data, training and UMAP](notebooks/gastrulation/mouseGastrulation.ipynb)
  - [Pretrained full-data models: balanced and unbalanced predictions in the original UMAP](notebooks/gastrulation/pretrainedModels.ipynb)
- [Held-out biological reconstruction](notebooks/06_heldout_biological.ipynb)
- [Gastrulation dynamics and mass decomposition](notebooks/07_dynamics.ipynb)
- [Cross-modal map training and quality checks](notebooks/08_map_quality.ipynb)
- [Lineage gene and peak dynamics](notebooks/09_lineage_features.ipynb)

- [scMultisim](notebooks/scMultisim/)

## TraInfBench

[TraInfBench](https://github.com/shankswang953/TraInfBench) hosts the external-method benchmark code used to compare COATI with CytoBridge, TrajectoryNet, MIOFlow, TIGON and scMultiNODE. It includes method-specific running instructions, evaluation and figure-analysis scripts, and dataset information.
