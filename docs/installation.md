# Installation

Use Python 3.10 or newer in a fresh environment. From the COATI repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks,docs,dev,anndata]'
```

On Windows activate with `.venv\Scripts\activate`. For training only, install `-e .`. AnnData and notebook support are optional extras; the core numerical dependencies install with the package. Import name: `coati`. Distribution name: `coati-trajectory`. No PyPI release is currently assumed.

## CPU and GPU

The public default is `device="cpu"`. Use `Config(device="cuda:0")` for a compatible CUDA environment, or `device="mps"` explicitly on supported Macs. `device="auto"` preserves the original CUDA/MPS/CPU selection. Only CPU has been checked in this release.

## Documentation locally

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

Open `http://localhost:8000`. `.readthedocs.yaml` provides a build configuration for connecting this repository to Read the Docs; creating an account project is a separate hosting step.

## Existing conda environments

Prefer the fresh environment above. Mixed conda/pip OpenMP runtimes can conflict on macOS. COATI does not set `KMP_DUPLICATE_LIB_OK`, change global tensor defaults or silently select a different solver to mask environment issues.
