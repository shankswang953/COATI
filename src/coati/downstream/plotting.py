"""Small publication-style plotting helpers for reusable downstream tables."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def _embedding_coordinates(values, transform, name):
    array = np.asarray(values, dtype=float)
    if array.ndim not in (2, 3) or not len(array):
        raise ValueError(f"{name} must be a non-empty 2D or 3D array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    original_shape = array.shape
    flat = array.reshape(-1, original_shape[-1])
    if transform is None:
        if flat.shape[1] != 2:
            raise ValueError(
                f"{name} has {flat.shape[1]} features. Supply an existing embedding transform."
            )
        embedded = flat
    else:
        function = transform.transform if hasattr(transform, "transform") else transform
        if not callable(function):
            raise TypeError("transform must be callable or expose a transform method.")
        embedded = np.asarray(function(flat), dtype=float)
        if embedded.shape != (len(flat), 2) or not np.isfinite(embedded).all():
            raise ValueError("transform must return one finite two-dimensional coordinate per row.")
    return embedded.reshape(*original_shape[:-1], 2)


def _selection_mask(selection, size, name):
    if selection is None:
        return np.ones(size, dtype=bool)
    values = np.asarray(selection)
    if values.dtype == bool:
        if values.ndim != 1 or len(values) != size:
            raise ValueError(f"{name} boolean mask must have length {size}.")
        return values.copy()
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must contain boolean values or integer indices.")
    if np.any(values < 0) or np.any(values >= size):
        raise IndexError(f"{name} contains an index outside [0, {size}).")
    mask = np.zeros(size, dtype=bool)
    mask[values.astype(int)] = True
    return mask


def plot_trajectory_embedding(
    trajectory,
    *,
    reference=None,
    reference_embedding=None,
    transform=None,
    labels=None,
    focus_labels=None,
    reference_subset=None,
    trajectory_subset=None,
    view="global",
    times=None,
    ax=None,
    max_paths=300,
    max_reference=20_000,
    random_state=0,
    cmap="viridis",
    path_color=None,
    path_alpha=0.45,
    linewidth=0.8,
    reference_color="#9CA3AF",
    reference_size=5,
    reference_alpha=0.28,
    show_endpoints=True,
    show_time_colorbar=True,
    padding=0.05,
    title=None,
    xlabel="UMAP 1",
    ylabel="UMAP 2",
):
    """Plot trajectories in an existing two-dimensional embedding.

    Parameters
    ----------
    trajectory
        Array with shape ``(n_times, n_cells, n_features)``. Pass already
        embedded coordinates, or provide the fitted reducer/callable through
        ``transform``. This function never fits an embedding.
    reference, reference_embedding
        Optional reference cells in feature space or their existing 2D
        coordinates. ``reference_embedding`` takes precedence and avoids
        transforming the reference data again.
    labels, focus_labels
        Optional reference-cell labels and values used to define a local
        biological region. In global view, non-focused cells remain visible as
        context. In local view, only focused reference cells determine the view.
    reference_subset, trajectory_subset
        Boolean masks or integer indices for explicit reference-cell and path
        selection. They can be combined with ``focus_labels``.
    view
        ``"global"`` preserves the full reference extent. ``"local"`` zooms to
        the selected reference cells and trajectories.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot. Projected trajectory coordinates and the
        plotted path indices are available as ``ax.coati_trajectory_embedding``
        and ``ax.coati_path_indices``.
    """
    raw_trajectory = np.asarray(trajectory, dtype=float)
    if raw_trajectory.ndim != 3 or raw_trajectory.shape[0] < 2 or not raw_trajectory.shape[1]:
        raise ValueError("trajectory must have shape (n_times >= 2, n_cells > 0, n_features).")
    embedded = _embedding_coordinates(raw_trajectory, transform, "trajectory")
    n_times, n_cells, _ = embedded.shape

    if view not in {"global", "local"}:
        raise ValueError("view must be 'global' or 'local'.")
    if not isinstance(max_paths, int) or isinstance(max_paths, bool) or max_paths < 1:
        raise ValueError("max_paths must be a positive integer.")
    if not isinstance(max_reference, int) or isinstance(max_reference, bool) or max_reference < 1:
        raise ValueError("max_reference must be a positive integer.")
    if not 0 <= padding < 1:
        raise ValueError("padding must lie in [0, 1).")

    time_values = np.arange(n_times, dtype=float) if times is None else np.asarray(times, dtype=float)
    if time_values.shape != (n_times,) or not np.isfinite(time_values).all():
        raise ValueError(f"times must contain {n_times} finite values.")

    path_mask = _selection_mask(trajectory_subset, n_cells, "trajectory_subset")
    path_indices = np.flatnonzero(path_mask)
    if not len(path_indices):
        raise ValueError("trajectory_subset selects no paths.")
    rng = np.random.default_rng(random_state)
    if len(path_indices) > max_paths:
        path_indices = np.sort(rng.choice(path_indices, size=max_paths, replace=False))
    paths = embedded[:, path_indices]

    background = None
    background_mask = None
    focus_mask = None
    has_reference_focus = reference_subset is not None or focus_labels is not None
    if reference_embedding is not None:
        background = _embedding_coordinates(reference_embedding, None, "reference_embedding")
        if background.ndim != 2:
            raise ValueError("reference_embedding must be a cells by 2 matrix.")
    elif reference is not None:
        background = _embedding_coordinates(reference, transform, "reference")
        if background.ndim != 2:
            raise ValueError("reference must be a cells by features matrix.")
    if background is not None:
        background_mask = _selection_mask(reference_subset, len(background), "reference_subset")
        if focus_labels is not None:
            if labels is None:
                raise ValueError("labels are required when focus_labels is used.")
            label_values = np.asarray(labels)
            if label_values.ndim != 1 or len(label_values) != len(background):
                raise ValueError("labels must contain one value per reference cell.")
            requested = np.atleast_1d(focus_labels)
            focus_mask = np.isin(label_values, requested)
            background_mask &= focus_mask
        if not background_mask.any():
            raise ValueError("The reference selection is empty.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 5.2))
    if background is not None:
        context_mask = np.ones(len(background), dtype=bool) if view == "global" else background_mask
        context_indices = np.flatnonzero(context_mask)
        if len(context_indices) > max_reference:
            context_indices = np.sort(rng.choice(context_indices, size=max_reference, replace=False))
        ax.scatter(
            background[context_indices, 0], background[context_indices, 1],
            s=reference_size, c=reference_color, alpha=reference_alpha,
            linewidths=0, rasterized=True, zorder=0,
        )
        if view == "global" and has_reference_focus:
            selected_indices = np.flatnonzero(background_mask)
            if len(selected_indices) > max_reference:
                selected_indices = np.sort(rng.choice(selected_indices, size=max_reference, replace=False))
            ax.scatter(
                background[selected_indices, 0], background[selected_indices, 1],
                s=reference_size * 1.3, c=reference_color,
                alpha=min(1.0, reference_alpha * 2.2), linewidths=0,
                rasterized=True, zorder=1,
            )

    segments = np.stack([paths[:-1], paths[1:]], axis=3).transpose(0, 1, 3, 2).reshape(-1, 2, 2)
    if path_color is None:
        segment_times = np.repeat((time_values[:-1] + time_values[1:]) / 2, len(path_indices))
        collection = LineCollection(
            segments, cmap=cmap, linewidths=linewidth, alpha=path_alpha, zorder=2,
        )
        collection.set_array(segment_times)
        collection.set_clim(time_values.min(), time_values.max())
    else:
        collection = LineCollection(
            segments, colors=path_color, linewidths=linewidth, alpha=path_alpha, zorder=2,
        )
    ax.add_collection(collection)
    if show_endpoints:
        ax.scatter(paths[0, :, 0], paths[0, :, 1], s=12, c="#2563EB", linewidths=0, zorder=3)
        ax.scatter(paths[-1, :, 0], paths[-1, :, 1], s=12, c="#F59E0B", linewidths=0, zorder=3)
    if path_color is None and show_time_colorbar:
        colorbar = ax.figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label("Time")

    if view == "global" and background is not None:
        extent = background
    elif background is not None:
        extent = np.concatenate([background[background_mask], paths.reshape(-1, 2)])
    else:
        extent = paths.reshape(-1, 2)
    span = np.ptp(extent, axis=0)
    fallback = max(float(np.max(span)), 1.0)
    margin = np.where(span > 0, span, fallback) * padding
    ax.set_xlim(extent[:, 0].min() - margin[0], extent[:, 0].max() + margin[0])
    ax.set_ylim(extent[:, 1].min() - margin[1], extent[:, 1].max() + margin[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.coati_trajectory_embedding = embedded
    ax.coati_path_indices = path_indices
    return ax


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })


def plot_terminal_vs_trajectory(
    df: pd.DataFrame,
    terminal_col: str,
    trajectory_col: str,
    out_path: str,
    hue_col: str | None = None,
    title: str | None = None,
) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    if hue_col is None:
        ax.scatter(df[terminal_col], df[trajectory_col], s=6, c="#4B5563", alpha=0.28, linewidths=0)
    else:
        groups = list(pd.unique(df[hue_col]))
        cmap = plt.get_cmap("tab10")
        for i, group in enumerate(groups):
            sub = df[df[hue_col].eq(group)]
            ax.scatter(sub[terminal_col], sub[trajectory_col], s=6, alpha=0.35, linewidths=0, color=cmap(i % 10), label=str(group))
        ax.legend(frameon=False, markerscale=2)
    if len(df) > 2:
        r = np.corrcoef(df[terminal_col], df[trajectory_col])[0, 1]
        ax.text(0.98, 0.98, f"r = {r:.2f}", transform=ax.transAxes, ha="right", va="top")
    ax.set_xlabel("terminal difference")
    ax.set_ylabel("trajectory difference")
    if title:
        ax.set_title(title)
    ax.grid(True, color="#E5E7EB", linewidth=0.7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    if out_path.endswith(".png"):
        fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)


def plot_metric_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    out_path: str,
    title: str | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
) -> None:
    set_publication_style()
    mat = df.pivot(index=row_col, columns=col_col, values=value_col)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.42 * mat.shape[1]), max(3.0, 0.35 * mat.shape[0])))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([str(x) for x in mat.columns], rotation=0)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([str(x) for x in mat.index])
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    if title:
        ax.set_title(title)
    if annotate:
        vals = mat.to_numpy()
        threshold = np.nanmean(vals)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    color = "white" if vals[i, j] < threshold else "black"
                    ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    if out_path.endswith(".png"):
        fig.savefig(out_path.replace(".png", ".pdf"))
    plt.close(fig)
