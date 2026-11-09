"""Evaluate the four trained gastrulation LOSO predictions in RNA and ATAC space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from coati.downstream import evaluate_paired_distributions


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = (
    ("balanced", "E8.0", ROOT / "outputs/gastrulation-balanced-loo-e8.0"),
    ("unbalanced", "E8.0", ROOT / "outputs/gastrulation-unbalanced-loo-e8.0"),
    ("balanced", "E8.5", ROOT / "outputs/gastrulation-balanced-loo-e8.5"),
    ("unbalanced", "E8.5", ROOT / "outputs/gastrulation-unbalanced-loo-e8.5"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data/gastrulation_embeddings.h5ad",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/gastrulation-loso-metrics",
    )
    parser.add_argument("--max-clusters", type=int, default=512)
    parser.add_argument("--support-k", type=int, default=15)
    parser.add_argument("--sinkhorn-blur", type=float, default=0.05)
    args = parser.parse_args()

    cells = ad.read_h5ad(args.data, backed="r")
    rows = []
    for model, stage, directory in DEFAULT_RUNS:
        metadata = json.loads((directory / "result.json").read_text())
        with np.load(directory / "heldout_prediction.npz", allow_pickle=False) as prediction:
            stage_time = float(prediction["time"])
            held_out = np.isclose(cells.obs["time"].to_numpy(dtype=float), stage_time)
            labels = cells.obs.loc[held_out, "celltype"].astype(str).to_numpy()
            paired_metrics = evaluate_paired_distributions(
                np.asarray(prediction["primary"], dtype=np.float32)
                / float(metadata["primary_scale"]),
                np.asarray(cells.obsm["X_pca"][held_out], dtype=np.float32)
                / float(metadata["primary_scale"]),
                labels,
                np.asarray(prediction["secondary"], dtype=np.float32)
                / float(metadata["secondary_scale"]),
                np.asarray(cells.obsm["X_atac"][held_out], dtype=np.float32)
                / float(metadata["secondary_scale"]),
                labels,
                log_weights=(
                    np.asarray(prediction["log_weights"], dtype=np.float64)
                    if "log_weights" in prediction.files
                    else None
                ),
                support_k=args.support_k,
                max_clusters=args.max_clusters,
                sinkhorn_blur=args.sinkhorn_blur,
                unknown_label="unknown",
            )
        metrics = {
            key.replace("primary_", "rna_").replace("secondary_", "atac_"): value
            for key, value in paired_metrics.items()
        }
        rows.append({"model": model, "held_out_stage": stage, **metrics})

    table = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "six_metrics.csv", index=False)
    (args.output / "evaluation.json").write_text(
        json.dumps(
            {
                "evaluation_space": {
                    "rna": "globally normalized RNA PCA50",
                    "atac": "globally normalized ATAC LSI14",
                },
                "support_recall": {"real_data_neighbors": args.support_k},
                "sinkhorn_w2": {
                    "definition": "sqrt(2 * debiased Sinkhorn divergence)",
                    "p": 2,
                    "blur": args.sinkhorn_blur,
                    "max_weighted_microclusters": args.max_clusters,
                },
                "composition_similarity": "1 - sqrt(base-2 JSD), excluding unknown",
                "mass": "uniform for balanced; normalized native particle mass for unbalanced",
                "total_population_size_evaluated": False,
            },
            indent=2,
        )
        + "\n"
    )
    display_columns = [
        "model",
        "held_out_stage",
        "rna_support_recall",
        "atac_support_recall",
        "rna_sinkhorn_w2",
        "atac_sinkhorn_w2",
        "rna_composition_similarity",
        "atac_composition_similarity",
    ]
    print(table[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {args.output / 'six_metrics.csv'}")


if __name__ == "__main__":
    main()
