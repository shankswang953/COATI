"""Read lineage gene/peak dynamics using the functions in tutorial 09.

The short example may return no accepted Somitic lineage. Diagnostic curves
remain explicitly labeled NMP-origin and retain readout validation warnings.
"""
import argparse
from pathlib import Path
import os
import uuid
import json
import hashlib
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
    parser.add_argument("--model", required=True, help="A saved paired gastrulation COATI result directory")
    parser.add_argument("--output", required=True, help="A fresh directory")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    project_dir = Path(__file__).resolve().parents[1]
    LATENT_DIR, RAW_DIR, MODEL_DIR = Path(args.latent_dir), Path(args.features_dir), Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    genes = ["T", "Sox2", "Tbx6"]
    peaks = ["chr3:34653105-34653705", "chr7:126785067-126785667", "chr7:126785767-126786367"]


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
    lineage.assignments.to_csv(output_dir / "lineage_assignments.csv", index=False)
    print("Accepted NMP → Somitic paths:", len(lineage.indices), "/", len(initial_barcodes))
    print("Empirical terminal support distance:", lineage.support_cutoff)
    lineage.assignments.groupby("assigned_celltype").agg(paths=("path", "size"), selected=("selected", "sum"))


    cache_path = project_dir / "outputs/lineage-nmp-features/observed_features.npz"
    signature_path = cache_path.with_name("source_signature.json")
    source_signature = {str((RAW_DIR / name).resolve()): {"size": (RAW_DIR / name).stat().st_size,
                        "mtime_ns": (RAW_DIR / name).stat().st_mtime_ns}
                        for name in ["gastrulation_rna_processed.h5ad", "gastrulation_atac_processed.h5ad"]}
    if cache_path.exists() and signature_path.exists() and json.loads(signature_path.read_text()) == source_signature:
        # This local cache was extracted from the named source files by the same reader.
        with np.load(cache_path, allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached["barcodes"], cells.obs_names.to_numpy(dtype=str))
            np.testing.assert_array_equal(cached["genes"], genes)
            np.testing.assert_array_equal(cached["peaks"], peaks)
            gene_values, peak_values = cached["rna"].copy(), cached["atac"].copy()
    else:
        gene_source = ds.read_h5ad_features(RAW_DIR / "gastrulation_rna_processed.h5ad", genes, obs_names=cells.obs_names)
        peak_source = ds.read_h5ad_features(RAW_DIR / "gastrulation_atac_processed.h5ad", peaks, obs_names=cells.obs_names, binary=True)
        np.testing.assert_array_equal(gene_source.barcodes, peak_source.barcodes)
        gene_values, peak_values = gene_source.values, peak_source.values
    print("Observed genes / peaks:", gene_values.shape, peak_values.shape)


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
    quality.to_csv(output_dir / "readout_validation.csv", index=False)
    print(quality.round(4).to_string(index=False))


    weighting = "growth" if "log_weights" in trajectory else "uniform"
    if len(lineage.indices):
        lineage_genes = ds.compute_feature_dynamics(rna_decoder, trajectory, selection=lineage, weighting=weighting)
        lineage_peaks = ds.compute_feature_dynamics(atac_decoder, trajectory, modality="secondary", selection=lineage, weighting=weighting)
        ds.plot_feature_dynamics(lineage_genes)
        plt.close("all")
        ds.plot_feature_dynamics(lineage_peaks, color="#e69f00")
        plt.close("all")
        lineage_genes.summary.to_csv(output_dir / "lineage_gene_dynamics.csv", index=False)
        lineage_peaks.summary.to_csv(output_dir / "lineage_peak_dynamics.csv", index=False)
    else:
        print("No accepted NMP → Somitic lineage in this checkpoint. No lineage-specific molecular curves are reported.")


    diagnostic_genes = ds.compute_feature_dynamics(rna_decoder, trajectory, weighting=weighting)
    diagnostic_peaks = ds.compute_feature_dynamics(atac_decoder, trajectory, modality="secondary", weighting=weighting)
    reference_times = [1., 2., 2.5]
    reference_types = ["NMP", "Somitic mesoderm", "Somitic mesoderm"]
    for dynamics, observed, color, filename in [
        (diagnostic_genes, gene_values, "#0072b2", "diagnostic_genes"),
        (diagnostic_peaks, peak_values, "#e69f00", "diagnostic_peaks"),
    ]:
        axes = ds.plot_feature_dynamics(dynamics, color=color)
        for j, ax in enumerate(axes):
            reference_means = [observed[np.isclose(cells.obs["time"], t) & (cells.obs["celltype"] == label), j].mean()
                               for t, label in zip(reference_times, reference_types)]
            ax.scatter(reference_times, reference_means, color="#243445", marker="D", s=30, label="Observed annotation-cohort mean")
            ax.lines[0].set_label("Model NMP-origin mean")
            ax.set_xticks(reference_times, ["E8.0", "E8.5", "E8.75"])
            if dynamics.binary and peak_quality.set_index("feature").loc[dynamics.features[j], "skill_vs_training_mean"] <= 0:
                ax.set_title(dynamics.features[j] + " · readout did not beat baseline", fontsize=10)
        axes[0].legend(frameon=False, fontsize=8, loc="upper right")
        axes[0].figure.suptitle(f"NMP-origin diagnostic · {len(lineage.indices)} accepted Somitic paths", fontsize=12)
        axes[0].figure.savefig(output_dir / (filename + ".png"), dpi=180, bbox_inches="tight")
        axes[0].figure.savefig(output_dir / (filename + ".pdf"), bbox_inches="tight")
        plt.close("all")
        dynamics.summary.to_csv(output_dir / (filename + ".csv"), index=False)


    ds.plot_feature_heatmap(diagnostic_genes, standardize=True)
    plt.title("NMP-origin gene readout · diagnostic")
    plt.gcf().savefig(output_dir / "diagnostic_gene_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    np.savez_compressed(output_dir / "trajectory.npz", **trajectory, initial_barcodes=initial_barcodes)
    report = {
        "requested_lineage": "E8.0 NMP → E8.75 Somitic mesoderm",
        "initial_paths": len(initial_barcodes), "accepted_paths": len(lineage.indices),
        "model_directory": str(MODEL_DIR.resolve()), "model_iterations": result.metadata["config"]["niters"],
        "lineage_threshold": .6, "terminal_support_cutoff": lineage.support_cutoff,
        "readout_test_cells": len(test_rows), "readout_validation": quality.astype(object).where(pd.notna(quality), None).to_dict(orient="records"),
        "gene_support": diagnostic_genes.summary.groupby("time").coverage.first().tolist(),
        "peak_support": diagnostic_peaks.summary.groupby("time").coverage.first().tolist(),
        "weighting": weighting, "genes": genes, "peaks": peaks,
        "molecular_readout": "Distance-weighted interpolation; binary ATAC targets give local open probabilities",
        "scope": "Current short checkpoint has no accepted Somitic branch; NMP-origin diagnostic curves are not lineage validation or evidence of priming.",
    }
    (output_dir / "downstream_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Output:", output_dir.resolve())


if __name__ == "__main__":
    main()
