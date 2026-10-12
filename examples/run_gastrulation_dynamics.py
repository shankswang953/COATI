"""Train a short paired gastrulation model and plot stage-specific UMAP dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import torch
import coati as ct


def prepare_inputs(data_dir):
    data_dir = Path(data_dir)
    rna = ad.read_h5ad(data_dir / "rna.h5ad")
    atac = ad.read_h5ad(data_dir / "atac.h5ad")
    if not rna.obs_names.is_unique or not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA and ATAC require unique, identically ordered cell barcodes.")
    np.testing.assert_array_equal(rna.obs["time_point_processed"], atac.obs["time_point_processed"])
    if rna.obsm["X_latent"].shape[1] != 50 or atac.obsm["X_latent"].shape[1] != 14:
        raise ValueError("This example expects the prepared RNA50 / ATAC14 gastrulation inputs.")
    missing_labels = int(rna.obs["celltype"].isna().sum())
    labels = rna.obs["celltype"].astype(object).fillna("unknown")
    cells = ct.prepare_anndata(
        rna, use_rep="X_latent", time_key="time_point_processed", celltype=labels,
        secondary=atac.obsm["X_latent"], secondary_key="X_atac", warn=False,
    )
    np.testing.assert_array_equal(cells.obsm["X_latent"], rna.obsm["X_latent"])
    np.testing.assert_array_equal(cells.obsm["X_umap"], rna.obsm["X_umap"])
    if not cells.obs_names.equals(rna.obs_names):
        raise AssertionError("Input preparation changed cell order.")
    return cells, missing_labels


def plot_stage(result, stage_cells):
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    names = stage_cells.obs["celltype"].cat.categories
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    palette = dict(zip(names, colors))
    palette["unknown"] = "#b5bac2"
    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    result.plot_dynamics(stage_cells, axes=axes, labels=False, palette=palette,
                         point_size=5, alpha=.75, density=1.3, linewidth=.65,
                         arrow_size=.8, grid_size=60)
    legend = axes[0].get_legend()
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]
    legend.remove()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015),
               ncol=4, frameon=False, fontsize=9, title="Cell type", markerscale=1.3)
    stage = str(stage_cells.obs["stage"].iloc[0])
    fig.suptitle(f"Mouse gastrulation · {stage} · {stage_cells.n_obs:,} cells", fontsize=18, y=.96)
    axes[0].set_position([.04, .30, .40, .60])
    axes[1].set_position([.52, .30, .40, .60])
    fig.axes[2].set_position([.935, .36, .012, .47])
    fig.savefig(result.output_dir / "dynamics_umap.png", dpi=180, bbox_inches="tight")
    fig.savefig(result.output_dir / "dynamics_umap.pdf", bbox_inches="tight")
    return axes


def validate_and_store(result, cells, stage_cells, data_dir, missing_labels):
    output = result.output_dir
    stored = result.write_dynamics(stage_cells, copy=True)
    assert "coati_velocity" not in cells.obsm and "coati_growth" not in cells.obs
    assert stored.obs_names.equals(stage_cells.obs_names)
    for values in (stored.obsm["coati_velocity"], stored.obsm["coati_velocity_umap"], stored.obs["coati_growth"]):
        assert np.isfinite(values).all()
    stored.write_h5ad(output / "dynamics.h5ad")
    reread = ad.read_h5ad(output / "dynamics.h5ad")
    np.testing.assert_array_equal(reread.obsm["X_umap"], stage_cells.obsm["X_umap"])
    assert reread.obs_names.equals(stage_cells.obs_names)
    np.testing.assert_array_equal(reread.obs["coati_growth"], stored.obs["coati_growth"])
    np.testing.assert_array_equal(reread.obsm["coati_velocity"], stored.obsm["coati_velocity"])
    np.testing.assert_array_equal(reread.obsm["coati_velocity_umap"], stored.obsm["coati_velocity_umap"])
    restored = ct.load_result(output)
    points, times = stage_cells.obsm["X_latent"], stage_cells.obs["time"].to_numpy()
    np.testing.assert_array_equal(result.compute_velocity(points, time=times), restored.compute_velocity(points, time=times))
    np.testing.assert_array_equal(result.compute_growth(points, time=times), restored.compute_growth(points, time=times))
    checkpoint = torch.load(output / "checkpoints/path_param.pth", map_location="cpu", weights_only=False)
    rates = stored.obs["coati_growth"].to_numpy()
    report = dict(dataset="gastrulation", accession="GSE205117", date="2026-10-12",
        source_sha256={name: hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest() for name in ("rna.h5ad", "atac.h5ad")},
        training_cells=cells.n_obs, primary_dim=50, secondary_dim=14,
        stages=cells.obs.groupby("stage", observed=True).size().to_dict(),
        times=result.data.times.tolist(), plot_stage=str(stage_cells.obs["stage"].iloc[0]), plot_cells=stage_cells.n_obs,
        missing_annotations_as_unknown=missing_labels, checkpoint_iterations=checkpoint["iteration"],
        config=result.metadata["config"], map_report=json.loads((output / "map/report.json").read_text()),
        growth_definition="d log(weight) / dt, including alpha_growth", growth_units="per model time unit",
        growth_range=[float(rates.min()), float(rates.max())], growth_quantiles=np.quantile(rates, [.05,.5,.95]).tolist(),
        positive_growth_fraction=float((rates > 0).mean()),
        velocity_projection="Local regularized projection from RNA50 to existing source UMAP at each cell's observed time",
        normalization="Reuse X_latent as supplied; no additional normalization", save_reload_agrees=True,
        stored_h5ad_roundtrip_agrees=True, finite_vectors_and_growth=True,
        scope="One-seed short unbalanced multimodal run; no convergence, proliferation or lineage-accuracy claim")
    (output / "dynamics_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Prepared gastrulation folder containing rna.h5ad and atac.h5ad")
    parser.add_argument("--output", required=True, help="A new directory for training, or an existing result with --reuse")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--stage", default="E8.5")
    parser.add_argument("--reuse", action="store_true", help="Reload the saved model instead of training again")
    args = parser.parse_args()
    cells, missing = prepare_inputs(args.data_dir)
    if args.stage not in cells.obs["stage"].unique():
        raise ValueError(f"Unknown stage: {args.stage}")
    if args.reuse:
        result = ct.load_result(args.output)
        for i, t in enumerate(result.data.times):
            np.testing.assert_array_equal(result.data.primary[i], cells.obsm["X_latent"][cells.obs["time"] == t])
            np.testing.assert_array_equal(result.data.secondary[i], cells.obsm["X_atac"][cells.obs["time"] == t])
    else:
        config = ct.Config(multimodal=True, balanced=False, iterations=args.iterations,
                           batch_size=128, hidden_dim=128, secondary_weight=.35,
                           manifold_weight=.1, sync_density_loss=True, seed=0)
        result = ct.train_vector_field(cells, config, output_dir=args.output)
    stage_cells = cells[cells.obs["stage"] == args.stage].copy()
    axes = plot_stage(result, stage_cells)
    plt.close(axes[0].figure)
    report = validate_and_store(result, cells, stage_cells, args.data_dir, missing)
    print(json.dumps({key: report[key] for key in ("checkpoint_iterations", "plot_stage", "plot_cells", "growth_range", "save_reload_agrees")}, indent=2))
    print("Figure:", result.output_dir / "dynamics_umap.png")


if __name__ == "__main__":
    main()
