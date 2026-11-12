"""Train a strict palate leave-one-stage-out COATI model."""
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
STAGE_TO_TIME = {"E13.5": 1.0, "E14.0": 1.5}
STAGE_TO_MAP = {
    "E13.5": DATA / "T_FiLM_lsi15_LOTO_holdout_1.pt",
    "E14.0": DATA / "T_FiLM_lsi15_LOTO_holdout_2.pt",
}


def main(*, default_config="balanced_loso.json", run_name="palate-balanced-loso"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "palate_embeddings.h5ad")
    parser.add_argument("--config", type=Path, default=HERE / default_config)
    parser.add_argument("--held-out-stage", choices=tuple(STAGE_TO_TIME), default="E13.5")
    parser.add_argument("--map-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs") / run_name)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()

    settings = json.loads(args.config.read_text())["training"]
    if args.iterations is not None:
        settings["iterations"] = args.iterations
    if args.device is not None:
        settings["device"] = args.device
    cells = ct.prepare_anndata(
        ad.read_h5ad(args.data), use_rep="X_pca", time_key="time",
        celltype_key="celltype", barcode_key="barcode",
        secondary_key="X_atac", warn=False,
    )
    full_data = ct.TemporalData.from_anndata(cells, secondary_obsm_key="X_atac")
    held_out_time = STAGE_TO_TIME[args.held_out_stage]
    training_data = full_data.without_times([held_out_time])
    checkpoint = args.map_checkpoint or STAGE_TO_MAP[args.held_out_stage]
    result = ct.fit(
        training_data, ct.Config(**settings), output_dir=args.output,
        T_model=load_film_checkpoint(checkpoint),
        primal_norm_path=DATA / "primal_norm_params.pt",
        sec_norm_path=DATA / "secondary_norm_params_lsi15.pt",
    )
    n_steps = int(round((held_out_time - full_data.times[0]) / 0.1))
    prediction_times = np.linspace(full_data.times[0], held_out_time, n_steps + 1)
    prediction = result.predict(times=prediction_times)
    secondary = result.project_secondary(prediction)
    payload = {"time": held_out_time, "primary": prediction["primary"][-1],
               "secondary": secondary[-1]}
    if "log_weights" in prediction:
        payload["log_weights"] = prediction["log_weights"][-1]
    np.savez_compressed(result.output_dir / "heldout_prediction.npz", **payload)
    print("Model:", result.output_dir)
    print("Held-out prediction:", result.output_dir / "heldout_prediction.npz")


if __name__ == "__main__":
    main()
