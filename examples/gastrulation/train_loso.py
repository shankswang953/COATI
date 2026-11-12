"""Train strict gastrulation leave-one-stage-out COATI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import coati as ct

from .film import load_film_checkpoint


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STAGES = {
    "E8.0": {
        "time": 1.0,
        "map": "T_FiLM_LOTO_holdout_1.pt",
        "mass_prior": [1.0, 6.0, 11.0],
    },
    "E8.5": {
        "time": 2.0,
        "map": "T_FiLM_LOTO_holdout_2.pt",
        "mass_prior": [1.0, 4.0, 11.0],
    },
}


def main(*, default_config="balanced_loso.json", run_name="gastrulation-balanced-loso"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "gastrulation_embeddings.h5ad")
    parser.add_argument("--config", type=Path, default=HERE / default_config)
    parser.add_argument("--held-out-stage", choices=tuple(STAGES), default="E8.0")
    parser.add_argument("--map-checkpoint", type=Path)
    parser.add_argument("--primal-norm", type=Path, default=DATA / "primal_norm_params.pt")
    parser.add_argument("--secondary-norm", type=Path, default=DATA / "secondary_norm_params.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs") / run_name)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()

    settings = json.loads(args.config.read_text())["training"]
    if args.iterations is not None:
        settings["iterations"] = args.iterations
    if args.device is not None:
        settings["device"] = args.device
    held_out = STAGES[args.held_out_stage]
    if not settings["balanced"]:
        settings.setdefault("extra", {})["biological_num"] = held_out["mass_prior"]

    cells = ct.prepare_anndata(
        ad.read_h5ad(args.data), use_rep="X_pca", time_key="time",
        celltype_key="celltype", barcode_key="barcode",
        secondary_key="X_atac", warn=False,
    )
    full_data = ct.TemporalData.from_anndata(cells, secondary_obsm_key="X_atac")
    training_data = full_data.without_times([held_out["time"]])
    checkpoint = args.map_checkpoint or DATA / held_out["map"]
    result = ct.fit(
        training_data, ct.Config(**settings), output_dir=args.output,
        T_model=load_film_checkpoint(checkpoint),
        primal_norm_path=args.primal_norm,
        sec_norm_path=args.secondary_norm,
    )

    n_steps = int(round((held_out["time"] - full_data.times[0]) / 0.1))
    prediction_times = np.linspace(full_data.times[0], held_out["time"], n_steps + 1)
    prediction = result.predict(times=prediction_times)
    secondary = result.project_secondary(prediction)
    payload = {
        "stage": args.held_out_stage,
        "time": held_out["time"],
        "primary": prediction["primary"][-1],
        "secondary": secondary[-1],
    }
    if "log_weights" in prediction:
        payload["log_weights"] = prediction["log_weights"][-1]
    np.savez_compressed(result.output_dir / "heldout_prediction.npz", **payload)
    print("Model:", result.output_dir)
    print("Held-out prediction:", result.output_dir / "heldout_prediction.npz")


if __name__ == "__main__":
    main()
