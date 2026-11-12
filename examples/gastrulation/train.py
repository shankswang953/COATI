"""Load paired gastrulation data, train COATI and plot one UMAP flow view."""
import argparse
import json
from pathlib import Path
import uuid
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import coati as ct

from .film import load_film_checkpoint


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load_data(path):
    cells = ad.read_h5ad(path)
    if not cells.obs_names.is_unique:
        raise ValueError("Gastrulation barcodes must be unique.")
    return ct.prepare_anndata(
        cells, use_rep="X_pca", time_key="time", celltype_key="celltype",
        barcode_key="barcode", secondary_key="X_atac", warn=False,
    )


def plot_umap(result, cells, stage):
    if stage not in cells.obs["stage"].unique():
        raise ValueError(f"Unknown stage: {stage}")
    stage_cells = cells[cells.obs["stage"] == stage].copy()
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    palette = dict(zip(stage_cells.obs["celltype"].cat.categories, colors))
    palette["unknown"] = "#b5bac2"
    fig, ax = plt.subplots(figsize=(11, 9))
    result.plot_velocity(
        stage_cells, color="celltype", style="stream", palette=palette,
        point_size=5, alpha=.75, density=1.2, linewidth=.6, arrow_size=.8,
        ax=ax, title=f"Mouse gastrulation · {stage}",
    )
    legend = ax.get_legend()
    fig.legend(legend.legend_handles, [t.get_text() for t in legend.get_texts()],
               loc="lower center", ncol=4, frameon=False, fontsize=8.5, markerscale=1.2)
    legend.remove()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=.06, right=.98, bottom=.27, top=.94)
    return fig


def main(*, default_config="config.json", run_name="gastrulation"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "gastrulation_embeddings.h5ad")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name(default_config))
    parser.add_argument("--map-checkpoint", type=Path, default=DATA / "T_FiLM.pt")
    parser.add_argument("--primal-norm", type=Path, default=DATA / "primal_norm_params.pt")
    parser.add_argument("--secondary-norm", type=Path, default=DATA / "secondary_norm_params.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--stage")
    parser.add_argument("--reuse", action="store_true", help="Reload --output instead of training again")
    args = parser.parse_args()
    if args.reuse and args.output is None:
        parser.error("--reuse requires --output")
    settings = json.loads(args.config.read_text())
    if args.iterations is not None:
        settings["training"]["iterations"] = args.iterations
    if args.device is not None:
        settings["training"]["device"] = args.device
    stage = args.stage or settings["plot_stage"]
    cells = load_data(args.data)
    if stage not in cells.obs["stage"].unique():
        parser.error(f"Unknown stage: {stage}")
    output = args.output or Path("outputs") / (run_name + "-" + uuid.uuid4().hex[:8])
    if args.reuse:
        result = ct.load_result(output)
        for i, time in enumerate(result.data.times):
            rows = cells.obs["time"].to_numpy() == time
            np.testing.assert_array_equal(result.data.primary[i], cells.obsm["X_pca"][rows])
            np.testing.assert_array_equal(result.data.secondary[i], cells.obsm["X_atac"][rows])
    else:
        config = ct.Config(**settings["training"])
        T_model = load_film_checkpoint(args.map_checkpoint) if config.multimodal else None
        result = ct.train_vector_field(
            cells, config, output_dir=output,
            T_model=T_model, primal_norm_path=args.primal_norm,
            sec_norm_path=args.secondary_norm if config.multimodal else None,
        )
    fig = plot_umap(result, cells, stage)
    fig.savefig(result.output_dir / "umap_flow.png", dpi=180, bbox_inches="tight")
    fig.savefig(result.output_dir / "umap_flow.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Model:", result.output_dir)
    print("Figure:", result.output_dir / "umap_flow.png")


if __name__ == "__main__":
    main()
