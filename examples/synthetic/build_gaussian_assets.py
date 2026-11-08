"""Generate the shared Gaussian-bump AnnData and fitted cross-modal map."""
import argparse
import json
from pathlib import Path

import torch

from .gaussian_fresh import generate_fresh_anndata
from .gaussian_map import evaluate_bump_map, fit_bump_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-steps", type=int, default=6000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = {"observed.h5ad", "map_pairs.h5ad", "T.pt", "map_report.json"}
    existing = {path.name for path in args.output.iterdir()}
    if existing & expected:
        raise FileExistsError("Shared Gaussian assets already exist; choose a new directory.")

    torch.set_num_threads(1)
    cells, pairs = generate_fresh_anndata()
    model, history, fit = fit_bump_map(pairs, steps=args.map_steps)
    metrics, grid = evaluate_bump_map(model, pairs)

    cells.write_h5ad(args.output / "observed.h5ad", compression="gzip")
    pairs.write_h5ad(args.output / "map_pairs.h5ad", compression="gzip")
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": {
            "d_in": 2, "d_out": 3, "hidden": 160, "n_layers": 4,
            "num_frequencies": 10, "max_frequency": 18.0,
            "xy_residual_scale": 0.05,
        },
        "fit": fit,
    }, args.output / "T.pt")
    report = {
        "data": {
            "observed_cells": int(cells.n_obs),
            "map_pairs": int(pairs.n_obs),
            "initial_covariance": "0.004 I_2",
            "terminal_covariances": ["0.003 I_2", "0.003 I_2"],
            "observed_counts": [1000, 2000],
            "target_mass_ratio": 2.0,
        },
        "map_fit": fit,
        "map_metrics": metrics.reset_index().to_dict(orient="records"),
        "grid_z_correlation": grid["z_correlation"],
        "center_predicted_height": grid["center_predicted_height"],
        "training_history": history.to_dict(orient="records"),
    }
    (args.output / "map_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Shared synthetic assets:", args.output.resolve())


if __name__ == "__main__":
    main()
