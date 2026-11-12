"""Train a full-data pancreas COATI model from the prepared embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import coati as ct

from .film import load_film_checkpoint


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main(*, default_config="balanced.json", run_name="moscot-balanced"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "moscot_embeddings.h5ad")
    parser.add_argument("--config", type=Path, default=HERE / default_config)
    parser.add_argument("--map-checkpoint", type=Path, default=DATA / "T_FiLM_poissonvi.pt")
    parser.add_argument("--primal-norm", type=Path, default=DATA / "primal_norm_params.pt")
    parser.add_argument("--secondary-norm", type=Path,
                        default=DATA / "secondary_norm_params_poissonvi.pt")
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
    data = ct.TemporalData.from_anndata(cells, secondary_obsm_key="X_atac")
    result = ct.fit(
        data, ct.Config(**settings), output_dir=args.output,
        T_model=load_film_checkpoint(args.map_checkpoint),
        primal_norm_path=args.primal_norm,
        sec_norm_path=args.secondary_norm,
    )
    print("Model:", result.output_dir)


if __name__ == "__main__":
    main()
