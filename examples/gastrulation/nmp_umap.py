"""Fit reproducible RNA and ATAC UMAPs for NMP-related gastrulation cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import umap
from umap import UMAP


EXPECTED_UMAP_VERSION = "0.5.9.post2"
NMP_CELLTYPES = (
    "Caudal epiblast",
    "Caudal neurectoderm",
    "NMP",
    "Spinal cord",
    "Caudal Mesoderm",
    "Paraxial mesoderm",
    "Somitic mesoderm",
)
COLORS = {
    "Caudal epiblast": "#5559C6",
    "Caudal neurectoderm": "#6D9671",
    "NMP": "#AA5100",
    "Spinal cord": "#FBBE31",
    "Caudal Mesoderm": "#2D6D8A",
    "Paraxial mesoderm": "#D76D9A",
    "Somitic mesoderm": "#39C6FF",
}
RNA_UMAP_PARAMETERS = {
    "n_neighbors": 100,
    "min_dist": 0.8,
    "n_components": 2,
    "metric": "euclidean",
    "random_state": 42,
    "transform_seed": 42,
    "init": "spectral",
    "spread": 1.0,
    "n_jobs": 1,
    "angular_rp_forest": False,
}
ATAC_UMAP_PARAMETERS = {
    "n_neighbors": 150,
    "min_dist": 0.98,
    "n_components": 2,
    "metric": "euclidean",
    "random_state": 0,
    "transform_seed": 42,
    "init": "spectral",
    "spread": 1.0,
    "n_jobs": 1,
    "angular_rp_forest": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_nmp_related(cells: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Return the stable row mask and labels used by the published NMP UMAPs."""
    labels = cells.obs["celltype"].astype(str).to_numpy()
    mask = np.isin(labels, NMP_CELLTYPES)
    if int(mask.sum()) != 7931:
        raise ValueError(f"Expected 7,931 NMP-related cells, found {int(mask.sum()):,}.")
    return mask, labels[mask]


def fit_nmp_umaps(cells: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit both UMAPs from the stored embeddings without saving reducer objects."""
    if umap.__version__ != EXPECTED_UMAP_VERSION:
        raise RuntimeError(
            f"Exact coordinate reproduction requires umap-learn=={EXPECTED_UMAP_VERSION}; "
            f"found {umap.__version__}."
        )
    mask, labels = select_nmp_related(cells)
    rna = np.asarray(cells.obsm["X_pca"][mask], dtype=np.float32)
    atac = np.asarray(cells.obsm["X_atac"][mask], dtype=np.float32)
    if rna.shape != (7931, 50) or atac.shape != (7931, 14):
        raise ValueError(f"Unexpected embeddings: RNA {rna.shape}, ATAC {atac.shape}.")
    rna_umap = UMAP(**RNA_UMAP_PARAMETERS).fit_transform(rna)
    atac_umap = UMAP(**ATAC_UMAP_PARAMETERS).fit_transform(atac)
    return mask, labels, rna_umap, atac_umap


def plot_umaps(labels: np.ndarray, rna_umap: np.ndarray, atac_umap: np.ndarray):
    """Draw the paired NMP-related UMAP reference views."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for ax, coordinates, title in zip(axes, (rna_umap, atac_umap), ("RNA PCA50", "ATAC LSI14")):
        for celltype in NMP_CELLTYPES:
            keep = labels == celltype
            ax.scatter(
                coordinates[keep, 0], coordinates[keep, 1],
                s=5, color=COLORS[celltype], alpha=0.62, linewidths=0,
                rasterized=True, label=celltype,
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles, labels_out = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", ncol=4, frameon=False, markerscale=2.5)
    fig.suptitle("NMP-related mouse gastrulation cells")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parent / "data/gastrulation_embeddings.h5ad",
    )
    parser.add_argument("--output", type=Path, default=root / "outputs/gastrulation-nmp-umap")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cells = ad.read_h5ad(args.data)
    mask, labels, rna_umap, atac_umap = fit_nmp_umaps(cells)
    np.savez_compressed(
        args.output / "nmp_umap_coordinates.npz",
        observation_names=cells.obs_names.to_numpy()[mask],
        celltype=labels,
        rna_umap=rna_umap,
        atac_umap=atac_umap,
    )
    report = {
        "data": str(args.data.resolve()),
        "data_sha256": _sha256(args.data),
        "umap_learn_version": umap.__version__,
        "n_cells": int(mask.sum()),
        "celltypes": list(NMP_CELLTYPES),
        "row_order": "original AnnData order after stable cell-type filtering",
        "rna_parameters": RNA_UMAP_PARAMETERS,
        "atac_parameters": ATAC_UMAP_PARAMETERS,
        "saved_reducer": False,
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    fig = plot_umaps(labels, rna_umap, atac_umap)
    fig.savefig(args.output / "nmp_umap.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output / "nmp_umap.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
