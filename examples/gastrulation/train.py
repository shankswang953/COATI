"""Load paired gastrulation data, train COATI and plot one UMAP flow view."""
import argparse
import json
from pathlib import Path
import uuid
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import coati as ct


def load_data(directory):
    directory = Path(directory)
    rna = ad.read_h5ad(directory / "rna.h5ad")
    atac = ad.read_h5ad(directory / "atac.h5ad")
    if not rna.obs_names.is_unique or not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC must have unique, identically ordered barcodes.")
    np.testing.assert_array_equal(rna.obs["stage"], atac.obs["stage"])
    np.testing.assert_array_equal(rna.obs["time_point_processed"], atac.obs["time_point_processed"])
    return ct.prepare_anndata(
        rna, use_rep="X_latent", time_key="time_point_processed",
        celltype=rna.obs["celltype"].astype(object).fillna("unknown"),
        secondary=atac.obsm["X_latent"], secondary_key="X_atac", warn=False,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder containing rna.h5ad and atac.h5ad")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--stage")
    parser.add_argument("--reuse", action="store_true", help="Reload --output instead of training again")
    args = parser.parse_args()
    if args.reuse and args.output is None:
        parser.error("--reuse requires --output")
    settings = json.loads(args.config.read_text())
    if args.iterations is not None:
        settings["training"]["iterations"] = args.iterations
    stage = args.stage or settings["plot_stage"]
    cells = load_data(args.data_dir)
    if stage not in cells.obs["stage"].unique():
        parser.error(f"Unknown stage: {stage}")
    output = args.output or Path("outputs") / ("gastrulation-" + uuid.uuid4().hex[:8])
    if args.reuse:
        result = ct.load_result(output)
        for i, time in enumerate(result.data.times):
            rows = cells.obs["time"].to_numpy() == time
            np.testing.assert_array_equal(result.data.primary[i], cells.obsm["X_latent"][rows])
            np.testing.assert_array_equal(result.data.secondary[i], cells.obsm["X_atac"][rows])
    else:
        result = ct.train_vector_field(cells, ct.Config(**settings["training"]), output_dir=output)
    fig = plot_umap(result, cells, stage)
    fig.savefig(result.output_dir / "umap_flow.png", dpi=180, bbox_inches="tight")
    fig.savefig(result.output_dir / "umap_flow.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Model:", result.output_dir)
    print("Figure:", result.output_dir / "umap_flow.png")


if __name__ == "__main__":
    main()
