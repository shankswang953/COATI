# Installation and training on Mac and Linux

COATI supports Python 3.10 or newer. Python 3.11 is a recommended starting point
for a new environment. Use a separate environment for each project.

## 1. Create an environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If your Python executable has a different name, use that executable instead of
`python3.11`. Run the installation commands with the same Python that will run COATI.

## 2. Install PyTorch for your machine

On a Mac, install the supported PyTorch wheel for your Python and hardware:

```bash
python -m pip install torch
```

On Linux without an NVIDIA GPU, install the CPU build:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA GPUs on Linux, select the appropriate pip command using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
The CUDA build must be compatible with your NVIDIA driver. COATI uses the
installed PyTorch build; it does not install a system CUDA toolkit or alter driver
and library-path settings.

## 3. Install COATI

For Python scripts and the core API:

```bash
python -m pip install "coati-trajectory @ git+https://github.com/shankswang953/COATI.git"
```

For AnnData and UMAP plotting, use the `visualization` extra:

```bash
python -m pip install "coati-trajectory[visualization] @ git+https://github.com/shankswang953/COATI.git"
```

For the example notebooks and local development, clone the repository:

```bash
git clone https://github.com/shankswang953/COATI.git
cd COATI
python -m pip install -e '.[notebooks,visualization]'
```

The import name is `coati`; the distribution name is `coati-trajectory`.
Repository examples and notebooks are kept outside the installed package.
Wheel and source distributions contain the library rather than the example
datasets, research checkpoints or notebook outputs. Clone the repository to use
those examples. A Git-based pip installation also requires Git.

## 4. Verify the installation

```bash
python -m pip check
coati --version
python -c "import coati as ct; print(ct.resolve_device('auto'))"
```

This verifies imports and device detection without downloading data or training.
To exercise the training and saving workflow on small CPU data:

```python
import numpy as np
import coati as ct

rng = np.random.default_rng(0)
initial = rng.normal(0, 0.05, (128, 2))
terminal = rng.normal(0, 0.05, (128, 2)) + 0.5
data = ct.TemporalData([initial, terminal], times=[0.0, 1.0])
config = ct.Config(iterations=5)
result = ct.fit(data, config, device="cpu", output_dir="outputs/install-check")
loaded = ct.load_result(result.output_dir, device="cpu")
```

Choose a fresh output directory when repeating this check. Existing run files
are never silently replaced.

## Choosing a device

Use `device="cpu"`, `"mps"`, `"cuda"`, `"cuda:0"`, or another visible CUDA index.
`device="auto"` preserves the existing CUDA → MPS → CPU priority. `Config()` keeps
its existing CPU default so old scripts do not switch backends unexpectedly.

```python
config = ct.Config(device="auto", iterations=200)
result = ct.train_vector_field(data, config, output_dir="outputs/run")

# A fit-level override uses a copy; it does not change config.
result = ct.fit(data, config, device="cpu", output_dir="outputs/cpu-run")
```

Explicit GPU requests fail early if the device is unavailable. They do not
silently fall back to the CPU. MPS requires a compatible Mac and PyTorch build;
[PyTorch's MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html) describe
the requirements. CUDA indices refer to devices visible to the current process.
Selecting a different backend does not change loss weights, preprocessing,
sampling, integration or optimizer settings. Results from different backends
are not guaranteed to be bitwise identical.

For the command-line training API, the device flag overrides the config:

```bash
coati fit snapshots.npz --keys t0 t1 --times 0 1 \
  --device auto --output outputs/run
```

Models can be reloaded on another available device using
`ct.load_result(path, device="cpu")` or `ct.load_result(path, device="auto")`.

## Finding results and diagnosing a failed run

Training prints the actual selected device and absolute output directory before
fitting. On success, it prints the final checkpoint and training-log paths.
With `verbose=True`, progress is shown live and also saved in `training.log`.
Without live progress, the same log retains training output.

A trajectory-training failure keeps its partial output directory, writes the
traceback to the training log and re-raises the original exception. Map quality
checks retain their diagnostics under `map/report.txt` and `map/report.json`.
No automatic retry,
batch-size reduction or device switch is performed. A failed run does not print
the successful-completion message. Correct the reported issue and choose a new
output directory for the next attempt.

The input snapshots must be finite and time ordered. `batch_size` cannot exceed
the smallest snapshot. The existing density calculation also requires
`density_topk <= batch_size`, even if its coefficient is zero; COATI reports this
before training rather than changing either value for you.
