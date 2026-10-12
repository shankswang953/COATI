"""Explicit AnnData conventions and conversion of prepared feature arrays."""
from pathlib import Path
import warnings
import numpy as np


class InputConventionWarning(UserWarning):
    """Input metadata was adapted to COATI's recommended AnnData fields."""


_ALIASES = {
    "time": ("time_point_processed", "processed_time", "time_index", "stage_num", "stage"),
    "celltype": ("cell_type", "cell_types", "cellType", "CellType", "annotation"),
    "barcode": ("cell_id", "cell_barcode", "barcodes", "Barcode"),
}


def resolve_representation(adata, use_rep="auto"):
    """Resolve an explicitly recorded representation, then common prepared keys."""
    if use_rep == "auto":
        recorded = adata.uns.get("coati_input", {}).get("use_rep")
        if recorded is not None:
            use_rep = None if recorded == "X" else recorded
        else:
            use_rep = next((key for key in ("X_pca", "X_latent", "X_coati") if key in adata.obsm), None)
    if use_rep is not None and use_rep not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{use_rep!r}]; select the representation used in training.")
    return use_rep


def observation_times(adata, *, time=None, time_key="time", time_mapping=None):
    """Resolve numeric model times without inferring a categorical chronology."""
    if time is not None:
        values = time
    else:
        if time_key not in adata.obs:
            raise KeyError(f"Missing adata.obs[{time_key!r}]; call prepare_anndata or supply time.")
        values = adata.obs[time_key].to_numpy()
        if time_mapping is not None:
            try:
                values = [time_mapping[label] for label in values]
            except KeyError as exc:
                raise ValueError(f"time_mapping is missing label {exc.args[0]!r}.") from exc
    try:
        values = np.broadcast_to(np.asarray(values, dtype=float), (adata.n_obs,)).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Times must be numeric per-cell values; categorical stages require time_mapping.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Times must be finite and use the same units as model training.")
    return values


def _column(obs, canonical, key):
    if key is not None:
        if key not in obs:
            raise KeyError(f"Missing observation column {key!r}.")
        return key
    if canonical in obs:
        return canonical
    candidates = [name for name in _ALIASES[canonical] if name in obs]
    if len(candidates) > 1:
        raise ValueError(f"Multiple candidates for {canonical}: {candidates}. Specify {canonical}_key explicitly.")
    return candidates[0] if candidates else None


def _per_cell(values, counts, name, *, stage_values=False):
    """Expand explicit snapshot times or flatten corresponding metadata vectors."""
    total = sum(counts)
    if isinstance(values, (list, tuple)) and len(values) == len(counts) and all(np.ndim(x) == 1 for x in values):
        if any(len(x) != n for x, n in zip(values, counts)):
            raise ValueError(f"{name} snapshot lengths must match input row counts.")
        values = np.concatenate(values)
    array = np.asarray(values)
    if stage_values and array.ndim == 1 and len(array) == len(counts):
        return np.repeat(array, counts)
    if array.ndim == 0:
        array = np.repeat(array, total)
    if array.shape != (total,):
        raise ValueError(f"{name} must provide one value per cell.")
    return array


def prepare_anndata(data, *, time=None, celltype=None, barcode=None, obs=None,
                    time_key=None, celltype_key=None, barcode_key=None, time_mapping=None,
                    use_rep="auto", secondary=None, secondary_key=None, umap=None,
                    compute_umap=False, n_neighbors=30, min_dist=.35, seed=0, copy=True,
                    warn=True):
    """Standardize AnnData or wrap prepared arrays without normalizing features.

    Recommended fields are obs['celltype'], obs['time'], obs['barcode'], with
    barcodes also in obs_names and UMAP in obsm['X_umap']. These are COATI
    conventions compatible with Scanpy, not names mandated by Scanpy.
    Existing source columns are retained. Copies are returned by default.

    Arrays may be a cells-by-features matrix, a list of snapshots, or TemporalData.
    Snapshot lists require explicit numeric times; TemporalData supplies its times
    and paired secondary snapshots. Missing labels become 'unknown', never inferred
    cell types. Missing array barcodes are generated. Ambiguous aliases require an
    explicit source-column key. Categorical stage labels require time_mapping.

    compute_umap=True fits a display UMAP on the selected prepared representation.
    Otherwise existing coordinates are reused; no embedding is fitted implicitly.
    """
    import anndata as ad
    import pandas as pd
    from .data import TemporalData, _array
    if not isinstance(copy, bool) or not isinstance(warn, bool) or not isinstance(compute_umap, bool):
        raise TypeError("copy, warn and compute_umap must be booleans.")
    if isinstance(data, (str, Path)):
        data = ad.read_h5ad(data)
    messages, generated = [], False
    counts = None
    if isinstance(data, ad.AnnData):
        if obs is not None:
            raise ValueError("obs is only accepted when constructing AnnData from arrays.")
        target = data.copy() if copy else data
        counts = [target.n_obs]
        rep = resolve_representation(target, use_rep)
    else:
        generated = True
        if isinstance(data, TemporalData):
            if time is None:
                time = data.times
            if secondary is None and data.secondary is not None:
                secondary = data.secondary
            parts = data.primary
        elif isinstance(data, (list, tuple)) and data and all((x.ndim if hasattr(x, "ndim") else np.ndim(x)) == 2 for x in data):
            parts = [_array(x) for x in data]
        else:
            parts = [_array(data)]
        if len({part.shape[1] for part in parts}) != 1:
            raise ValueError("Snapshot feature dimensions must agree.")
        counts = [len(part) for part in parts]
        target = ad.AnnData(np.concatenate(parts), obs=None if obs is None else obs.copy())
        rep = "X_coati" if use_rep == "auto" else use_rep
        if rep is not None:
            target.obsm[rep] = np.asarray(target.X).copy()
        if time is not None:
            time = _per_cell(time, counts, "time", stage_values=len(parts) > 1)
    if target.n_obs == 0:
        raise ValueError("AnnData must contain cells.")
    matrix = target.X if rep is None else target.obsm[rep]
    if getattr(matrix, "ndim", None) != 2 or matrix.shape[0] != target.n_obs or matrix.shape[1] == 0:
        raise ValueError("The selected representation must be cells by prepared features.")
    values = matrix.data if hasattr(matrix, "toarray") else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError("The selected representation must be finite.")

    source_time = None if time is not None else _column(target.obs, "time", time_key)
    if time is None and source_time is None:
        raise ValueError("Provide numeric time values or an observation time column; time cannot be inferred from features.")
    if time is not None and time_mapping is not None:
        explicit = np.broadcast_to(np.asarray(time), (target.n_obs,))
        try:
            time = np.asarray([time_mapping[label] for label in explicit], dtype=float)
        except KeyError as exc:
            raise ValueError(f"time_mapping is missing label {exc.args[0]!r}.") from exc
    times = observation_times(target, time=time, time_key=source_time, time_mapping=time_mapping)
    if source_time is not None and source_time != "time":
        messages.append(f"{source_time} -> time")
    source_type = None if celltype is not None else _column(target.obs, "celltype", celltype_key)
    if celltype is not None:
        labels = _per_cell(celltype, counts, "celltype")
    elif source_type is not None:
        labels = target.obs[source_type].copy()
        if source_type != "celltype":
            messages.append(f"{source_type} -> celltype")
    else:
        labels = np.repeat("unknown", target.n_obs)
        messages.append("missing cell types -> unknown")
    if pd.isna(labels).any():
        raise ValueError("celltype cannot contain missing values; provide 'unknown' explicitly for unlabeled cells.")

    source_barcode = None if barcode is not None else _column(target.obs, "barcode", barcode_key)
    if barcode is not None:
        ids = _per_cell(barcode, counts, "barcode")
    elif source_barcode is not None:
        ids = target.obs[source_barcode].to_numpy()
        if source_barcode != "barcode":
            messages.append(f"{source_barcode} -> barcode")
    elif generated and obs is None:
        ids = np.array([f"cell-{i:07d}" for i in range(target.n_obs)])
        messages.append("generated barcodes for array rows")
    else:
        ids = target.obs_names.to_numpy()
        messages.append("obs_names -> barcode")
    if pd.isna(ids).any():
        raise ValueError("barcodes must be non-missing and unique.")
    ids = pd.Index([str(value) for value in ids])
    if ids.has_duplicates or any(not value.strip() for value in ids):
        raise ValueError("barcodes must be nonempty and globally unique; do not silently deduplicate cell identities.")
    if not target.obs_names.equals(ids) and not generated:
        if "coati_original_obs_name" not in target.obs:
            target.obs["coati_original_obs_name"] = target.obs_names.to_numpy()
    if source_time == "time" and time_mapping is not None:
        target.obs["coati_original_time"] = target.obs["time"].copy()
    target.obs["time"] = times
    target.obs["celltype"] = pd.Categorical(labels)
    target.obs["barcode"] = ids.to_numpy()
    target.obs_names = ids
    if source_type is not None and f"{source_type}_colors" in target.uns:
        target.uns["celltype_colors"] = np.asarray(target.uns[f"{source_type}_colors"]).copy()

    if secondary is not None:
        secondary_key = secondary_key or "X_secondary"
        if isinstance(secondary, (list, tuple)) and len(secondary) == len(counts) and all((x.ndim if hasattr(x, "ndim") else np.ndim(x)) == 2 for x in secondary):
            if any(len(x) != n for x, n in zip(secondary, counts)):
                raise ValueError("Secondary snapshot row counts must match primary snapshots.")
            secondary = np.concatenate([_array(x) for x in secondary])
        secondary = _array(secondary)
        if len(secondary) != target.n_obs:
            raise ValueError("Secondary rows must match primary cell identities and order.")
        target.obsm[secondary_key] = secondary
    else:
        available = [key for key in ("X_secondary", "X_atac") if key in target.obsm]
        recorded = target.uns.get("coati_input", {}).get("secondary_rep")
        if secondary_key is not None:
            if secondary_key not in target.obsm:
                raise KeyError(f"Missing paired secondary representation {secondary_key!r}.")
        elif recorded in target.obsm:
            secondary_key = recorded
        elif len(available) == 1:
            secondary_key = available[0]
        else:
            secondary_key = ""
    if umap is not None:
        coordinates = _array(umap)
        if coordinates.shape != (target.n_obs, 2):
            raise ValueError("umap must contain two coordinates per cell in unchanged row order.")
        target.obsm["X_umap"] = coordinates
    if compute_umap:
        if umap is not None or "X_umap" in target.obsm:
            raise ValueError("Existing UMAP coordinates are present; omit compute_umap to reuse them.")
        if target.n_obs < 4:
            raise ValueError("UMAP fitting requires at least four cells.")
        if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 2:
            raise ValueError("n_neighbors must be an integer of at least two.")
        from umap import UMAP
        target.obsm["X_umap"] = UMAP(n_neighbors=min(n_neighbors, target.n_obs - 1),
            min_dist=min_dist, random_state=seed, transform_seed=seed, n_jobs=1).fit_transform(matrix)
    previous = target.uns.get("coati_input", {})
    target.uns["coati_input"] = dict(use_rep="X" if rep is None else rep,
        secondary_rep=secondary_key, time_key="time", celltype_key="celltype", barcode_key="barcode",
        generated_from_arrays=generated or bool(previous.get("generated_from_arrays", False)),
        umap_generated=compute_umap or bool(previous.get("umap_generated", False)))
    if messages and warn:
        warnings.warn("COATI recommends obs['celltype'], obs['time'], obs['barcode'] and obsm['X_umap']. "
                      + "; ".join(messages) + ". Source columns are retained; row order is unchanged.",
                      InputConventionWarning, stacklevel=2)
    return target
