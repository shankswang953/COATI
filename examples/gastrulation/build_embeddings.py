"""Build the lightweight paired gastrulation AnnData from prepared PCA/LSI assets."""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


KEYS = ("time0", "time1", "time2", "time3")
TIMES = (0.0, 1.0, 2.0, 2.5)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Directory containing the prepared gastrulation data")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "data/gastrulation_embeddings.h5ad")
    args = parser.parse_args()
    source = args.source.resolve()
    rna_path = source / "rna_pca_by_time.npz"
    atac_path = source / "atac_lsi_by_time_14D.npz"
    labels_path = source / "celltype_sub_by_stage.npz"
    obs_path = source / "gastrulation_rna_processed.h5ad"
    with np.load(rna_path, allow_pickle=False) as archive:
        rna = np.concatenate([archive[key] for key in KEYS]).astype(np.float32, copy=False)
    with np.load(atac_path, allow_pickle=False) as archive:
        atac = np.concatenate([archive[key] for key in KEYS]).astype(np.float32)
    with np.load(labels_path, allow_pickle=True) as archive:
        labels = np.concatenate([archive[key] for key in KEYS]).astype(str)
    source_cells = ad.read_h5ad(obs_path, backed="r")
    if not np.array_equal(np.asarray(source_cells.obsm["X_pca"])[:, :50], rna):
        raise ValueError("Processed RNA AnnData rows do not match rna_pca_by_time.npz.")
    if not np.array_equal(source_cells.obs["celltype"].astype(str).to_numpy(), labels):
        raise ValueError("Cell-type rows do not match celltype_sub_by_stage.npz.")
    obs = source_cells.obs[["barcode", "stage", "celltype", "processed_time"]].copy()
    obs["time"] = obs.pop("processed_time").astype(float)
    obs["celltype"] = obs["celltype"].astype(object).fillna("unknown")
    obs["barcode"] = source_cells.obs_names.astype(str)
    umap = np.asarray(source_cells.obsm["X_umap"]).astype(np.float32)
    source_cells.file.close()
    obs.index = pd.Index(obs["barcode"].to_numpy(), name=None)
    cells = ad.AnnData(rna.copy(), obs=obs)
    cells.obsm["X_pca"] = rna
    cells.obsm["X_atac"] = atac
    cells.obsm["X_umap"] = umap
    cells.uns["coati_input"] = {
        "use_rep": "X_pca", "secondary_rep": "X_atac", "time_key": "time",
        "celltype_key": "celltype", "barcode_key": "barcode", "generated_from_arrays": False,
        "umap_generated": False,
    }
    cells.uns["source_sha256"] = {
        "rna_pca_by_time.npz": sha256(rna_path),
        "atac_lsi_by_time_14D.npz": sha256(atac_path),
        "celltype_sub_by_stage.npz": sha256(labels_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cells.write_h5ad(args.output, compression="gzip")
    report = {
        "output": args.output.name,
        "cells": cells.n_obs,
        "rna_dimensions": rna.shape[1],
        "atac_dimensions": atac.shape[1],
        "times": list(TIMES),
        "rows_per_time": [int(np.sum(cells.obs["time"].to_numpy() == value)) for value in TIMES],
        "sha256": sha256(args.output),
        "sources": cells.uns["source_sha256"],
    }
    (args.output.parent / "gastrulation_embeddings.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
