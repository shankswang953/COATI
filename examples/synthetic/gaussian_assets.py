"""Shared data, map loading and growth plots for Gaussian-bump tutorials."""
from pathlib import Path
from urllib.request import urlretrieve

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from .gaussian_map import FourierBumpMLP
from .generate_gaussian_data import gaussian_bump_z


ASSET_NAMES = ("observed.h5ad", "map_pairs.h5ad", "T.pt", "map_report.json")
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/shankswang953/COATI/main/"
    "examples/synthetic/data/gaussian_bump"
)


def resolve_asset_directory(*, download_directory="outputs/synthetic-gaussian-data"):
    """Use repository assets when available, otherwise download public copies."""
    for root in (Path.cwd(), *Path.cwd().parents):
        candidate = root / "examples/synthetic/data/gaussian_bump"
        if candidate.is_dir():
            return candidate
    directory = Path(download_directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ASSET_NAMES:
        path = directory / name
        if not path.is_file():
            urlretrieve(f"{RAW_BASE_URL}/{name}", path)
    return directory


def load_shared_gaussian_assets(*, device="cpu", directory=None):
    """Load the committed AnnData and frozen Fourier map."""
    directory = Path(directory) if directory is not None else resolve_asset_directory()
    cells = ad.read_h5ad(directory / "observed.h5ad")
    payload = torch.load(directory / "T.pt", map_location=device, weights_only=True)
    model = FourierBumpMLP(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval().requires_grad_(False)
    return cells, model, payload


def plot_growth_along_trajectories(data, results, predictions):
    """Color every primary-space trajectory segment by instantaneous growth."""
    growth = {}
    for name, result in results.items():
        path = predictions[name]["primary"]
        times = np.broadcast_to(predictions[name]["times"][:, None], path.shape[:2])
        growth[name] = result.compute_growth(
            path.reshape(-1, path.shape[-1]), time=times.reshape(-1), raw=True
        ).reshape(path.shape[:2])
    limit = max(float(np.quantile(np.abs(values), .99)) for values in growth.values())
    limit = max(limit, 1e-8)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit)
    observed = np.concatenate(data.primary)
    grid = np.linspace(0., 1., 181)
    xx, yy = np.meshgrid(grid, grid)
    surface = gaussian_bump_z(np.stack([xx, yy], axis=-1))
    ncols = min(2, len(results))
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0, 4.3 * nrows),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    artist = None
    for ax, (name, result) in zip(axes, results.items()):
        path = predictions[name]["primary"].transpose(1, 0, 2)
        values = growth[name].T
        segments = np.stack([path[:, :-1], path[:, 1:]], axis=2).reshape(-1, 2, 2)
        segment_values = ((values[:, :-1] + values[:, 1:]) / 2.).reshape(-1)
        artist = LineCollection(segments, cmap="coolwarm", norm=norm,
                                linewidths=.6, alpha=.24, rasterized=True)
        artist.set_array(segment_values)
        ax.add_collection(artist)
        ax.scatter(observed[:, 0], observed[:, 1], s=3, color="#a8afb8",
                   alpha=.25, linewidths=0, rasterized=True)
        ax.contour(xx, yy, surface, levels=np.array([.05, .3, .7]),
                   colors="#343a40", linewidths=.8, alpha=.8)
        target = result.metadata["config"].get("biological_num", [1., 2.])[-1]
        predicted = np.exp(predictions[name]["log_weights"][-1]).sum()
        ax.set(title=f"{name}\nterminal mass {predicted:.2f} / {target:.2f}",
               xlabel="$x_1$", ylabel="$x_2$", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    for ax in axes[len(results):]:
        ax.remove()
    colorbar = fig.colorbar(artist, ax=axes, shrink=.82, pad=.02)
    colorbar.set_label("Instantaneous growth  $d\\log w / dt$")
    fig.suptitle("Growth along learned trajectories", fontsize=14)
    return fig, growth
