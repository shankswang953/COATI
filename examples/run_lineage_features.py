"""Plot gene/peak relative changes on one axis using tutorial 09 calculations.

Each temporal profile is standardized separately. Figure export is opt-in;
no trajectory archive, CSV tables or final report is automatically saved.
"""
import argparse
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import coati as ct
from coati import downstream as ds



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gene", default="Tbx6")
    parser.add_argument("--peak-region", default="promoter", help="Annotation region_type, or all")
    parser.add_argument("--max-peaks", type=int, default=2)
    parser.add_argument("--layout", choices=["combined", "separate"], default="combined")
    parser.add_argument("--show-observed", action="store_true")
    parser.add_argument("--show-iqr", action="store_true")
    parser.add_argument("--output", help="Optional fresh directory for figure export; no final report is saved")
    args = parser.parse_args()
    output_dir = Path(args.output) if args.output else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    project_dir = Path(__file__).resolve().parents[1]
    LATENT_DIR = Path(args.latent_dir)
    RAW_DIR = Path(args.features_dir)
    MODEL_DIR = Path(args.model)
    # Change the gene name; peak IDs are resolved automatically from annotations.
    GENE = args.gene
    PEAK_REGION = None if args.peak_region == "all" else args.peak_region  # None includes all candidate regions in the table.
    MAX_PEAKS = args.max_peaks             # None includes all eligible candidate peaks.
    PLOT_LAYOUT = args.layout # RNA and all peaks on one axis; "separate" uses separate figures.
    NORMALIZATION = "zscore" # Standardize each temporal profile independently.
    SHOW_IQR = args.show_iqr         # Optional across-path spread, in the same standardized units.
    SHOW_OBSERVED = args.show_observed     # Optional measured snapshot-cohort means.

    rna = ad.read_h5ad(LATENT_DIR / "rna.h5ad")
    atac = ad.read_h5ad(LATENT_DIR / "atac.h5ad")
    assert rna.obs_names.is_unique and rna.obs_names.equals(atac.obs_names)
    cells = ct.prepare_anndata(
        rna, use_rep="X_latent", time_key="time_point_processed",
        celltype=rna.obs["celltype"].astype(object).fillna("unknown"),
        secondary=atac.obsm["X_latent"], secondary_key="X_atac", warn=False,
    )
    result = ct.load_result(MODEL_DIR)
    for i, time in enumerate(result.data.times):
        mask = np.isclose(cells.obs["time"], time)
        np.testing.assert_array_equal(result.data.primary[i], cells.obsm["X_latent"][mask])
        np.testing.assert_array_equal(result.data.secondary[i], cells.obsm["X_atac"][mask])
    initial_mask = (cells.obs["stage"] == "E8.0") & (cells.obs["celltype"] == "NMP")
    trajectory = result.predict_trajectory(
        cells.obsm["X_latent"][initial_mask], start_time=1., times=np.linspace(1., 2.5, 16),
    )
    trajectory["secondary"] = result.project_secondary(trajectory)
    initial_barcodes = cells.obs_names[initial_mask].to_numpy(dtype=str)
    print("NMP starting cells:", len(initial_barcodes), "Start / end time:", trajectory["times"][[0, -1]])

    lineage = ds.select_lineage(
        trajectory, cells, terminal_celltypes="Somitic mesoderm",
        use_rep="X_latent", n_neighbors=15, min_probability=.6,
    )
    lineage.assignments.insert(1, "initial_barcode", initial_barcodes)
    print("Accepted NMP → Somitic paths:", len(lineage.indices), "/", len(initial_barcodes))
    print("Empirical terminal support distance:", lineage.support_cutoff)
    lineage.assignments.groupby("assigned_celltype").agg(paths=("path", "size"), selected=("selected", "sum"))

    links = ds.find_gene_peaks(
        GENE, RAW_DIR / "nmp_peak_gene_links.csv",
        region_type=PEAK_REGION, max_peaks=MAX_PEAKS,
    )
    genes, peaks = [GENE], links["peak"].tolist()

    cache_path = project_dir / "outputs/lineage-nmp-features/observed_features.npz"
    signature_path = cache_path.with_name("source_signature.json")
    source_signature = {str((RAW_DIR / name).resolve()): {"size": (RAW_DIR / name).stat().st_size,
                        "mtime_ns": (RAW_DIR / name).stat().st_mtime_ns}
                        for name in ["gastrulation_rna_processed.h5ad", "gastrulation_atac_processed.h5ad"]}
    gene_values = peak_values = None
    if cache_path.exists() and signature_path.exists() and json.loads(signature_path.read_text()) == source_signature:
        # Reuse only named columns present in this source-verified local cache.
        with np.load(cache_path, allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached["barcodes"], cells.obs_names.to_numpy(dtype=str))
            if set(genes) <= set(cached["genes"]) and set(peaks) <= set(cached["peaks"]):
                gene_columns = [list(cached["genes"]).index(name) for name in genes]
                peak_columns = [list(cached["peaks"]).index(name) for name in peaks]
                gene_values = cached["rna"][:, gene_columns].copy()
                peak_values = cached["atac"][:, peak_columns].copy()
    if gene_values is None:
        gene_source = ds.read_h5ad_features(RAW_DIR / "gastrulation_rna_processed.h5ad", genes, obs_names=cells.obs_names)
        peak_source = ds.read_h5ad_features(RAW_DIR / "gastrulation_atac_processed.h5ad", peaks, obs_names=cells.obs_names, binary=True)
        np.testing.assert_array_equal(gene_source.barcodes, peak_source.barcodes)
        gene_values, peak_values = gene_source.values, peak_source.values
    print("Observed genes / peaks:", gene_values.shape, peak_values.shape)
    links[["gene", "peak", "region_type", "distance"]]

    rng = np.random.default_rng(17)
    train_rows, test_rows = [], []
    for time in np.unique(cells.obs["time"]):
        rows = rng.permutation(np.flatnonzero(cells.obs["time"].to_numpy() == time))
        split = round(.2 * len(rows))
        train_rows.extend(rows[split:])
        test_rows.extend(rows[:split])
    train_rows, test_rows = np.array(train_rows), np.array(test_rows)
    assert not np.intersect1d(train_rows, test_rows).size
    rna_decoder = ds.fit_feature_decoder(
        cells.obsm["X_latent"][train_rows], gene_values[train_rows], features=genes,
        units="Source log-normalized expression",
    )
    atac_decoder = ds.fit_feature_decoder(
        cells.obsm["X_atac"][train_rows], peak_values[train_rows], features=peaks,
        binary=True, units="Local peak-open probability",
    )
    gene_quality = rna_decoder.validate(cells.obsm["X_latent"][test_rows], gene_values[test_rows])
    peak_quality = atac_decoder.validate(cells.obsm["X_atac"][test_rows], peak_values[test_rows])
    gene_quality = gene_quality.rename(columns={"MSE": "squared_error"})
    peak_quality = peak_quality.rename(columns={"Brier": "squared_error"})
    gene_quality["error_type"], peak_quality["error_type"] = "MSE", "Brier"
    gene_quality.insert(0, "modality", "RNA")
    peak_quality.insert(0, "modality", "ATAC")
    quality = pd.concat([gene_quality, peak_quality], ignore_index=True)
    quality.round(4)

    reference_times = [1., 2., 2.5]
    reference_types = ["NMP", "Somitic mesoderm", "Somitic mesoderm"]
    observed_rows = []
    for time, celltype in zip(reference_times, reference_types):
        mask = np.isclose(cells.obs["time"], time) & (cells.obs["celltype"] == celltype)
        if not mask.any():
            continue
        for modality, names, values in [("RNA", genes, gene_values), ("ATAC", peaks, peak_values)]:
            for j, feature in enumerate(names):
                observed_rows.append(dict(modality=modality, feature=feature, time=time,
                                          mean=float(values[mask, j].mean()), n_cells=int(mask.sum()), cohort=celltype))
    observed_summary = pd.DataFrame(observed_rows)
    weighting = "growth" if "log_weights" in trajectory else "uniform"
    if len(lineage.indices):
        lineage_pair = ds.compute_gene_peak_dynamics(
            GENE, trajectory, gene_decoder=rna_decoder, peak_decoder=atac_decoder,
            peak_gene_links=links, selection=lineage, weighting=weighting,
        )
        lineage_pair.plot(layout=PLOT_LAYOUT, normalization=NORMALIZATION,
                          show_observed=SHOW_OBSERVED, observed=observed_summary, show_iqr=SHOW_IQR)
        plt.show()
    else:
        print("No accepted NMP → Somitic lineage in this checkpoint. No lineage-specific molecular curves are reported.")

    diagnostic_pair = ds.compute_gene_peak_dynamics(
        GENE, trajectory, gene_decoder=rna_decoder, peak_decoder=atac_decoder,
        peak_gene_links=links, weighting=weighting,
    )

    def format_plot(axes_by_modality):
        unique_axes = list(dict.fromkeys([*axes_by_modality["gene"], *axes_by_modality["peaks"]]))
        for ax in unique_axes:
            ax.set_xticks(reference_times, ["E8.0", "E8.5", "E8.75"])
            ax.set_xlabel("Stage")
        for peak, ax in zip(diagnostic_pair.atac.features, axes_by_modality["peaks"]):
            if peak_quality.set_index("feature").loc[peak, "skill_vs_training_mean"] <= 0:
                for line in ax.lines:
                    if line.get_label().startswith(f"ATAC · {peak}"):
                        line.set_label(line.get_label() + " *")
        for ax in unique_axes:
            ax.legend(frameon=False, fontsize=9, loc="best")
        for figure in dict.fromkeys(ax.figure for ax in unique_axes):
            if len(figure.axes) == 1:
                figure.axes[0].set_title(f"{GENE} · NMP-origin diagnostic", fontsize=12)
            else:
                figure.suptitle(f"{GENE} · NMP-origin diagnostic", fontsize=12)
        if output_dir is not None:
            prefix = "combined" if len(unique_axes) == 1 else "separate"
            for index, figure in enumerate(dict.fromkeys(ax.figure for ax in unique_axes)):
                figure.savefig(output_dir / f"{prefix}_{index}.png", dpi=180, bbox_inches="tight")
                figure.savefig(output_dir / f"{prefix}_{index}.pdf", bbox_inches="tight")
        else:
            plt.show()

    axes = diagnostic_pair.plot(layout=PLOT_LAYOUT, normalization=NORMALIZATION,
                                show_observed=SHOW_OBSERVED, observed=observed_summary, show_iqr=SHOW_IQR)
    format_plot(axes)

    axes = diagnostic_pair.plot(layout="separate", normalization=NORMALIZATION,
                                show_observed=SHOW_OBSERVED, observed=observed_summary, show_iqr=SHOW_IQR)
    format_plot(axes)


if __name__ == "__main__":
    main()
