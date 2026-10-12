"""Prepare AnnData, train a toy model and inspect labeled UMAP motion and growth."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import coati as ct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    snapshots = ct.datasets.split(n=128, seed=7)
    features = np.concatenate(snapshots.primary)
    times = np.repeat(snapshots.times, [len(x) for x in snapshots.primary])
    labels = np.where(times == 0, "Initial", np.where(features[:, 1] < .5, "Branch A", "Branch B"))
    cells = ct.prepare_anndata(snapshots, celltype=labels, compute_umap=True, seed=7)
    result = ct.train_vector_field(cells, balanced=args.balanced, iterations=args.iterations,
                                  batch_size=64, seed=7, output_dir=args.output)
    axes = result.plot_dynamics(cells)
    axes[0].figure.savefig(result.output_dir / "dynamics_umap.png", dpi=180, bbox_inches="tight")
    axes[0].figure.savefig(result.output_dir / "dynamics_umap.pdf", bbox_inches="tight")
    plt.close(axes[0].figure)
    stored = result.write_dynamics(cells, copy=True)
    stored.write_h5ad(result.output_dir / "dynamics.h5ad")
    restored = ct.load_result(result.output_dir)
    np.testing.assert_array_equal(result.compute_velocity(), restored.compute_velocity())
    if not args.balanced:
        np.testing.assert_array_equal(result.compute_growth(), restored.compute_growth())
    report = {"balanced": args.balanced, "iterations": args.iterations,
              "fields": list(stored.obs.columns), "obsm": list(stored.obsm),
              "growth_definition": "d log(weight) / dt" if not args.balanced else "unavailable",
              "save_reload_agrees": True}
    (result.output_dir / "dynamics_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Figure:", result.output_dir / "dynamics_umap.png")


if __name__ == "__main__":
    main()
