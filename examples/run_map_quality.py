"""Train or reload T on gastrulation and check held-out paired-cell map quality.

Mirrors notebooks/08_map_quality.ipynb using the existing coati.maps API.
"""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import anndata as ad
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from umap import UMAP
import coati as ct
from coati.maps import train_map, load_map



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Prepared gastrulation rna.h5ad / atac.h5ad folder")
    parser.add_argument("--output", required=True, help="Fresh output folder, or an existing run with --reuse")
    parser.add_argument("--map-kind", choices=["film", "mlp"], default="film")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--reuse", action="store_true", help="Check the saved map without training or fitting UMAP again")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    DATA_DIR = Path(args.data_dir)
    output_dir = Path(args.output)
    map_kind = args.map_kind
    map_config = ct.MapConfig(epochs=args.epochs, patience=30)


    rna = ad.read_h5ad(DATA_DIR / "rna.h5ad")
    atac = ad.read_h5ad(DATA_DIR / "atac.h5ad")
    assert rna.obs_names.is_unique and rna.obs_names.equals(atac.obs_names)
    np.testing.assert_array_equal(rna.obs["time_point_processed"], atac.obs["time_point_processed"])
    assert rna.obsm["X_latent"].shape == (rna.n_obs, 50)
    assert atac.obsm["X_latent"].shape == (atac.n_obs, 14)
    cells = ct.prepare_anndata(
        rna, use_rep="X_latent", time_key="time_point_processed",
        celltype=rna.obs["celltype"].astype(object).fillna("unknown"),
        secondary=atac.obsm["X_latent"], secondary_key="X_atac", warn=False,
    )
    x, y = cells.obsm["X_latent"], cells.obsm["X_atac"]
    times = cells.obs["time"].to_numpy()
    stages = cells.obs["stage"].astype(str).to_numpy()
    labels = cells.obs["celltype"].astype(str).to_numpy()
    print("Paired cells:", cells.n_obs, "RNA:", x.shape, "ATAC:", y.shape)


    rng = np.random.default_rng(17)
    reference_parts, test_parts = [], []
    for time in np.unique(times):
        rows = rng.permutation(np.flatnonzero(times == time))
        n_test = round(.2 * len(rows))
        test_parts.append(rows[:n_test])
        reference_parts.append(rows[n_test:])
    reference_rows = np.sort(np.concatenate(reference_parts))
    test_rows = np.sort(np.concatenate(test_parts))
    assert not np.intersect1d(reference_rows, test_rows).size
    assert len(reference_rows) + len(test_rows) == cells.n_obs
    reference_cells = cells[reference_rows].copy()
    paired = ct.TemporalData.from_anndata(reference_cells, secondary_obsm_key="X_atac")
    print("Map fitting + validation:", len(reference_rows), "Independent test pairs:", len(test_rows))


    if args.reuse:
        prior = json.loads((output_dir / "quality_report.json").read_text())
        for name, digest in prior["source_sha256"].items():
            if hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Input differs from the saved run: {name}")
        if prior["map_kind"] != map_kind:
            raise ValueError("--map-kind must match the saved map.")
        map_config = ct.MapConfig(**prior["map_config"])
        T = load_map(output_dir / "map/model.pt", device="cpu")
        map_report = json.loads((output_dir / "map/report.json").read_text())
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        T, map_report = train_map(
            paired, kind=map_kind, config=map_config,
            directory=output_dir / "map", device="cpu", seed=0,
        )
    print("Map:", map_kind, "Best epoch:", map_report["best_epoch"],
          "Epochs completed:", map_report["epochs_completed"], "Validation:", map_report["status"])
    pd.DataFrame(map_report["stages"])[["time", "train_cells", "validation_cells", "normalized_mse", "sinkhorn_ratio"]]


    splits = np.load(output_dir / "map/splits.npz", allow_pickle=False)
    fit_parts, validation_parts = [], []
    for k, time in enumerate(paired.times):
        global_rows = reference_rows[times[reference_rows] == time]
        fit_parts.append(global_rows[splits[f"train_{k}"]])
        validation_parts.append(global_rows[splits[f"validation_{k}"]])
    fit_rows = np.concatenate(fit_parts)
    validation_rows = np.concatenate(validation_parts)
    assert not np.intersect1d(fit_rows, validation_rows).size
    assert not np.intersect1d(fit_rows, test_rows).size
    assert not np.intersect1d(validation_rows, test_rows).size
    assert set(np.concatenate([fit_rows, validation_rows, test_rows])) == set(range(cells.n_obs))

    def map_points(model, points, query_times):
        with torch.no_grad():
            return np.concatenate([
                model(torch.tensor(points[i:i+1024]), torch.tensor(query_times[i:i+1024], dtype=torch.float32)).numpy()
                for i in range(0, len(points), 1024)
            ])

    predicted = map_points(T, x[test_rows], times[test_rows])
    restored = load_map(output_dir / "map/model.pt", device="cpu")
    np.testing.assert_array_equal(predicted, map_points(restored, x[test_rows], times[test_rows]))
    assert predicted.shape == y[test_rows].shape and np.isfinite(predicted).all()
    print("Optimization / validation / test:", len(fit_rows), len(validation_rows), len(test_rows))
    print("Exact map checkpoint reload: passed")


    if args.reuse:
        atac_umap = joblib.load(output_dir / "atac_reference_umap.joblib")
        saved = np.load(output_dir / "test_predictions.npz", allow_pickle=False)
        np.testing.assert_array_equal(saved["test_rows"], test_rows)
        np.testing.assert_array_equal(saved["mapped_atac"], predicted)
        real_atac_umap, mapped_atac_umap = saved["real_umap"], saved["mapped_umap"]
    else:
        atac_umap = UMAP(n_neighbors=30, min_dist=.35, random_state=0, transform_seed=0, n_jobs=1)
        atac_umap.fit(y[fit_rows])
        real_atac_umap = atac_umap.transform(y[test_rows])
        mapped_atac_umap = atac_umap.transform(predicted)
    rna_umap = cells.obsm["X_umap"][test_rows]
    joblib.dump(atac_umap, output_dir / "atac_reference_umap.joblib")
    assert np.isfinite(real_atac_umap).all() and np.isfinite(mapped_atac_umap).all()

    names = list(cells.obs["celltype"].cat.categories)
    colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    palette = dict(zip(names, colors))
    palette["unknown"] = "#b5bac2"
    coordinates = [rna_umap, real_atac_umap, mapped_atac_umap]
    titles = ["Observed RNA", "Observed ATAC", "T(RNA, time) → ATAC" if map_kind == "film" else "T(RNA) → ATAC"]
    atac_limits = np.vstack([real_atac_umap, mapped_atac_umap])
    low, high = atac_limits.min(0), atac_limits.max(0)
    pad = .05 * np.maximum(high-low, 1.)

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for j, (ax, coords, title) in enumerate(zip(axes, coordinates, titles)):
        ax.scatter(*coords.T, c=[palette[label] for label in labels[test_rows]], s=4, alpha=.75, linewidths=0, rasterized=True)
        ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
        if j > 0:
            ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]), ylim=(low[1]-pad[1], high[1]+pad[1]))
    shown_names = [name for name in names if name in set(labels[test_rows])]
    handles = [Line2D([], [], marker="o", linestyle="", color=palette[name], label=name, markersize=5) for name in shown_names]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, title="Supplied cell type")
    fig.suptitle(f"Gastrulation · {len(test_rows):,} unseen paired cells", fontsize=17, y=.98)
    fig.subplots_adjust(left=.045, right=.99, bottom=.34, top=.86, wspace=.22)
    fig.savefig(output_dir / "map_celltype_umap.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "map_celltype_umap.pdf", bbox_inches="tight")
    plt.close(fig)


    stage_order = [str(cells.obs.loc[cells.obs["time"] == t, "stage"].iloc[0]) for t in np.unique(times)]
    stage_palette = dict(zip(stage_order, ["#0072b2", "#009e73", "#e69f00", "#cc79a7"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, (ax, coords, title) in enumerate(zip(axes, coordinates, titles)):
        ax.scatter(*coords.T, c=[stage_palette[s] for s in stages[test_rows]], s=4, alpha=.65, linewidths=0, rasterized=True)
        ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
        if j > 0:
            ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]), ylim=(low[1]-pad[1], high[1]+pad[1]))
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=stage_palette[s], label=s) for s in stage_order],
               loc="lower center", ncol=4, frameon=False, fontsize=12, title="Observation stage")
    fig.suptitle("The same test cells, colored by observation stage", fontsize=17, y=.98)
    fig.subplots_adjust(left=.045, right=.99, bottom=.20, top=.82, wspace=.22)
    fig.savefig(output_dir / "map_time_umap.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "map_time_umap.pdf", bbox_inches="tight")
    plt.close(fig)


    known_fit = fit_rows[labels[fit_rows] != "unknown"]
    known_test = labels[test_rows] != "unknown"
    celltype_reader = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1).fit(y[known_fit], labels[known_fit])
    stage_reader = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1).fit(y[fit_rows], stages[fit_rows])
    real_types = celltype_reader.predict(y[test_rows])
    mapped_types = celltype_reader.predict(predicted)
    real_stages = stage_reader.predict(y[test_rows])
    mapped_stages = stage_reader.predict(predicted)
    scored_types = np.unique(labels[test_rows][known_test])
    score_rows = []
    for name, type_pred, stage_pred in [("Real test ATAC", real_types, real_stages), ("Mapped test ATAC", mapped_types, mapped_stages)]:
        score_rows.append({
            "Input": name, "Annotated cells": int(known_test.sum()),
            "Celltype agreement (%)": 100 * accuracy_score(labels[test_rows][known_test], type_pred[known_test]),
            "Celltype macro recall (%)": 100 * recall_score(labels[test_rows][known_test], type_pred[known_test], labels=scored_types, average="macro", zero_division=0),
            "Stage agreement (%)": 100 * accuracy_score(stages[test_rows], stage_pred),
        })
    quality = pd.DataFrame(score_rows).set_index("Input")
    quality.to_csv(output_dir / "label_agreement.csv")
    print(quality.round(2).to_string())


    stage_rows = []
    for stage in stage_order:
        mask = stages[test_rows] == stage
        truth, estimate = y[test_rows][mask], predicted[mask]
        baseline_mean = y[fit_rows][stages[fit_rows] == stage].mean(0)
        mse = float(np.mean((estimate-truth)**2))
        baseline_mse = float(np.mean((baseline_mean-truth)**2))
        known = mask & known_test
        stage_rows.append({"Stage": stage, "Test pairs": int(mask.sum()), "Annotated pairs": int(known.sum()),
            "Paired RMSE": float(np.sqrt(mse)), "Normalized paired MSE": mse / baseline_mse,
            "Real celltype (%)": 100 * accuracy_score(labels[test_rows][known], real_types[known]),
            "Mapped celltype (%)": 100 * accuracy_score(labels[test_rows][known], mapped_types[known]),
            "Real stage (%)": 100 * accuracy_score(stages[test_rows][mask], real_stages[mask]),
            "Mapped stage (%)": 100 * accuracy_score(stages[test_rows][mask], mapped_stages[mask])})
    stage_quality = pd.DataFrame(stage_rows).set_index("Stage")
    stage_quality.to_csv(output_dir / "stage_quality.csv")
    print(stage_quality.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, predicted_stage, title in zip(axes, [real_stages, mapped_stages], ["Real test ATAC", "Mapped test ATAC"]):
        matrix = confusion_matrix(stages[test_rows], predicted_stage, labels=stage_order, normalize="true")
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{100*matrix[i,j]:.0f}%", ha="center", va="center", color="white" if matrix[i,j] > .5 else "#243445")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=stage_order, yticklabels=stage_order,
               xlabel="Reference classifier stage", ylabel="Supplied stage", title=title)
    fig.colorbar(image, ax=axes, shrink=.8, label="Row fraction")
    fig.savefig(output_dir / "stage_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


    type_rows = []
    for name in scored_types:
        mask = known_test & (labels[test_rows] == name)
        type_rows.append({"Cell type": name, "Test cells": int(mask.sum()),
                          "Real ATAC recall (%)": 100 * float(np.mean(real_types[mask] == name)),
                          "Mapped ATAC recall (%)": 100 * float(np.mean(mapped_types[mask] == name))})
    type_quality = pd.DataFrame(type_rows).set_index("Cell type")
    type_quality.to_csv(output_dir / "celltype_quality.csv")
    print(type_quality[type_quality["Test cells"] >= 20].sort_values("Mapped ATAC recall (%)").head(10).round(2).to_string())


    roles = np.full(cells.n_obs, "test", dtype="U12")
    roles[fit_rows] = "optimization"
    roles[validation_rows] = "validation"
    pd.DataFrame({"barcode": cells.obs_names, "stage": stages, "role": roles}).to_csv(output_dir / "cell_splits.csv", index=False)
    np.savez_compressed(output_dir / "test_predictions.npz", rna=x[test_rows], real_atac=y[test_rows], mapped_atac=predicted,
                        time=times[test_rows], celltype=labels[test_rows], barcode=cells.obs_names[test_rows].to_numpy(dtype=str),
                        test_rows=test_rows, real_umap=real_atac_umap, mapped_umap=mapped_atac_umap)
    joblib.dump({"celltype": celltype_reader, "stage": stage_reader}, output_dir / "reference_classifiers.joblib")
    report = {
        "dataset": "gastrulation", "accession": "GSE205117", "map_kind": map_kind,
        "seed": 0, "split_seed": 17, "input_dimensions": [50, 14],
        "source_sha256": {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() for name in ("rna.h5ad", "atac.h5ad")},
        "counts": {"optimization": len(fit_rows), "validation": len(validation_rows), "test": len(test_rows), "annotated_test": int(known_test.sum())},
        "map_config": map_config.to_dict(), "map_report": map_report,
        "quality": quality.reset_index().to_dict(orient="records"), "stage_quality": stage_quality.reset_index().to_dict(orient="records"),
        "celltype_quality": type_quality.reset_index().to_dict(orient="records"),
        "exact_map_reload": True, "disjoint_split_roles": True,
        "atac_umap_fit_rows": "optimization only", "atac_shared_transform": True,
        "classifier_reference": "real optimization ATAC14; 15 neighbors; distance weights",
        "time_is_map_input": map_kind == "film", "unknown_labels_excluded_from_celltype_scores": True,
        "scope": "Unseen paired cells within observed stages; precomputed source PCA/LSI and scales fixed. Stage consistency is conditional on supplied time for FiLM.",
    }
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved:", output_dir.resolve())
    print("Test pairs:", len(test_rows), "Annotated test pairs:", int(known_test.sum()))


if __name__ == "__main__":
    main()
