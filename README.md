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
src/coati/                 Python package and training engine
src/coati/downstream/      Reusable trajectory evaluation and downstream analyses
examples/                  Data preparation and standalone Python examples
notebooks/                 Executable tutorials
tests/                     Package tests
```

Dataset-specific notebook reproductions are maintained in [examples](examples/README.md), outside the installed package.

## Basic workflow

COATI requires time-indexed primary embeddings, such as RNA PCA, and optionally
paired secondary embeddings, such as ATAC LSI. For biological data, we recommend
an AnnData object with `time`, `celltype` and `barcode` annotations. A typical run
prepares the data, trains or loads the cross-modal map, fits the vector field, and
then performs trajectory inference and downstream analysis.

Run a short end-to-end example from the repository root:

```bash
python -m examples.synthetic.run_toy \
  --dataset gaussian \
  --niters 200 \
  --output outputs/gaussian-smoke
```

For a guided biological workflow, start the introductory notebook:

```bash
python -m jupyter lab notebooks/00_introduction.ipynb
```

## References

1. Wang, J., Zhao, Y., Nie, Q. and Cang, Z. [Synchronized Optimal Transport Reconstructs Coupled Transcriptomic and Chromatin Dynamics](https://arxiv.org/abs/2611.08853). arXiv:2611.08853, 2026.
2. Cang, Z., Wang, J., Wei, X. and Zhao, Y. [Synchronization of Unbalanced Dynamical Optimal Transport across Multiple Spaces](https://arxiv.org/abs/2602.18725). arXiv:2602.18725, 2026.
3. Cang, Z. and Zhao, Y. [Synchronized Optimal Transport for Joint Modeling of Dynamics Across Multiple Spaces](https://doi.org/10.1137/24M1667555). *SIAM Journal on Applied Mathematics* 85(1), 341–365, 2025.

## TraInfBench

[TraInfBench](https://github.com/shankswang953/TraInfBench) hosts the external-method benchmark code used to compare COATI with CytoBridge, TrajectoryNet, MIOFlow, TIGON and scMultiNODE. It includes method-specific running instructions, evaluation and figure-analysis scripts, and dataset information.
