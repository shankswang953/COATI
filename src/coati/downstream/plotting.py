"""Small publication-style plotting helpers for reusable downstream tables."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
