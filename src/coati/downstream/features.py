"""Lineage selection and local gene/peak readouts on model trajectories.

Readouts interpolate observed features in the corresponding latent space.
They are neither inverse PCA/LSI reconstructions nor molecular measurements.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _matrix(value, name):
    if hasattr(value, "toarray"):
        value = value.toarray()
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not min(value.shape) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite cells-by-features matrix.")
    return value


def _names(values):
    values = [values] if isinstance(values, str) else list(values)
    values = [str(v) for v in values]
    if not values or len(set(values)) != len(values):
        raise ValueError("Provide a nonempty list of distinct feature names.")
    return [str(v) for v in values]


def _feature_indices(var, features, feature_key):
    names = pd.Index(var.index if feature_key is None else var[feature_key].astype(str))
    indices = []
    for name in features:
        matches = np.flatnonzero(names == name)
        if len(matches) != 1:
            raise ValueError(f"Feature {name!r} has {len(matches)} matches; use an unambiguous feature ID.")
        indices.append(int(matches[0]))
    return np.asarray(indices)


@dataclass
class FeatureMatrix:
    """Selected observed features, with explicit cell identities and feature IDs."""
    values: np.ndarray
    features: tuple
    barcodes: np.ndarray
    feature_ids: tuple
    source: str
    binary: bool = False


def read_h5ad_features(path, features, *, obs_names=None, feature_key=None,
                       layer=None, use_raw=False, binary=False, block_size=256):
    """Read selected H5AD features in bounded blocks without loading the atlas.

    Supports dense, CSR and CSC matrices. Reorders cells by requested unique
    barcodes, rather than assuming RNA/ATAC file row order. binary=True converts
    nonnegative observed values to presence/absence before returning them.
    No normalization is applied. Raw gene counts and log-normalized expression
    require different explicit interpretation; select the appropriate layer.
    """
    import h5py
    from anndata.io import read_elem
    from scipy.sparse import csr_matrix
    names = _names(features)
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    if use_raw and layer is not None:
        raise ValueError("Choose use_raw or layer, not both.")
    with h5py.File(path, "r") as handle:
        obs = read_elem(handle["obs"])
        var = read_elem(handle["raw/var"] if use_raw else handle["var"])
        if not obs.index.is_unique:
            raise ValueError("Source observation barcodes must be unique.")
        wanted = obs.index if obs_names is None else pd.Index(obs_names)
        if not len(wanted) or not wanted.is_unique:
            raise ValueError("Requested observation barcodes must be nonempty and unique.")
        rows = obs.index.get_indexer(wanted)
        if np.any(rows < 0):
            raise ValueError(f"Missing source barcodes: {wanted[rows < 0].tolist()[:5]}")
        cols = _feature_indices(var, names, feature_key)
        key = "raw/X" if use_raw else ("X" if layer is None else f"layers/{layer}")
        node = handle[key]
        shape = node.shape if isinstance(node, h5py.Dataset) else tuple(node.attrs.get("shape", ()))
        if tuple(shape) != (len(obs), len(var)):
            raise ValueError("Selected matrix shape must match source observations and variables.")
        out = np.zeros((len(rows), len(cols)), dtype=np.float32)
        destination = np.full(len(obs), -1, dtype=np.int64)
        destination[rows] = np.arange(len(rows))
        if isinstance(node, h5py.Dataset):
            order = np.argsort(cols)
            inverse = np.argsort(order)
            for start in range(0, len(obs), block_size):
                stop = min(start + block_size, len(obs))
                chosen = destination[start:stop] >= 0
                if chosen.any():
                    block = node[start:stop, cols[order]][:, inverse]
                    out[destination[start:stop][chosen]] = block[chosen]
        else:
            encoding = node.attrs.get("encoding-type")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            # Older AnnData layers may omit encoding metadata; only unambiguous
            # indptr lengths permit inference of their sparse orientation.
            if encoding is None:
                length = len(node["indptr"])
                if len(obs) == len(var):
                    raise ValueError("Sparse layer orientation is ambiguous without encoding metadata.")
                encoding = "csr_matrix" if length == len(obs) + 1 else "csc_matrix" if length == len(var) + 1 else None
            if encoding == "csc_matrix":
                for j, col in enumerate(cols):
                    lo, hi = node["indptr"][col:col+2]
                    source_rows = node["indices"][int(lo):int(hi)]
                    values = node["data"][int(lo):int(hi)]
                    keep = destination[source_rows] >= 0
                    np.add.at(out[:, j], destination[source_rows[keep]], values[keep])
            elif encoding == "csr_matrix":
                for start in range(0, len(obs), block_size):
                    stop = min(start + block_size, len(obs))
                    chosen = destination[start:stop] >= 0
                    if not chosen.any():
                        continue
                    ptr = node["indptr"][start:stop+1]
                    lo, hi = int(ptr[0]), int(ptr[-1])
                    block = csr_matrix((node["data"][lo:hi], node["indices"][lo:hi], ptr-lo),
                                       shape=(stop-start, len(var)))
                    out[destination[start:stop][chosen]] = block[chosen][:, cols].toarray()
            else:
                raise ValueError(f"Unsupported H5AD matrix encoding: {encoding!r}.")
    if not np.isfinite(out).all():
        raise ValueError("Selected observed feature values must be finite.")
    if binary:
        if (out < 0).any():
            raise ValueError("Presence/absence requires nonnegative observed features.")
        out = (out > 0).astype(np.float32)
    return FeatureMatrix(out, tuple(names), wanted.to_numpy(dtype=str),
                         tuple(var.index[cols].astype(str)), str(Path(path).resolve()), binary)


def _neighbor_weights(distances):
    exact = distances <= 1e-12
    weights = 1 / np.maximum(distances, 1e-12)
    weights = np.where(exact.any(axis=1, keepdims=True), exact.astype(float), weights)
    return weights / weights.sum(axis=1, keepdims=True)


def _support_cutoff(reference, quantile, seed):
    if not 0 < quantile <= 1 or len(reference) < 2:
        raise ValueError("Support calibration needs two reference cells and a quantile in (0, 1].")
    rows = np.random.default_rng(seed).choice(len(reference), min(2048, len(reference)), replace=False)
    distances = NearestNeighbors(n_neighbors=2).fit(reference).kneighbors(reference[rows])[0][:, 1]
    return max(float(np.quantile(distances, quantile)), 1e-12)


@dataclass
class FeatureDecoder:
    """Distance-weighted local feature interpolation with an empirical support check."""
    reference: np.ndarray
    observed: np.ndarray
    features: tuple
    neighbor_model: object
    support_cutoff: float
    units: str
    binary: bool
    barcodes: np.ndarray

    def predict(self, points, *, max_distance=None, chunk_size=1024):
        """Return interpolated values, nearest distances and a supported-point mask.

        points may be cells-by-dimensions or time-by-paths-by-dimensions.
        Unsupported values are NaN. The cutoff is an empirical neighborhood
        screen, not a statistical confidence interval or guarantee of accuracy.
        Pass max_distance=np.inf only to explicitly disable this screen.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim not in (2, 3) or points.shape[-1] != self.reference.shape[1] or not points.size or not np.isfinite(points).all():
            raise ValueError("Points must be finite and match decoder reference dimensions.")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        cutoff = self.support_cutoff if max_distance is None else float(max_distance)
        if np.isnan(cutoff) or cutoff < 0:
            raise ValueError("max_distance must be nonnegative.")
        leading = points.shape[:-1]
        flat = points.reshape(-1, points.shape[-1])
        values, nearest = [], []
        for start in range(0, len(flat), chunk_size):
            distances, indices = self.neighbor_model.kneighbors(flat[start:start+chunk_size])
            weights = _neighbor_weights(distances)
            values.append(np.einsum("nk,nkf->nf", weights, self.observed[indices]))
            nearest.append(distances[:, 0])
        values, nearest = np.concatenate(values), np.concatenate(nearest)
        supported = nearest <= cutoff
        values[~supported] = np.nan
        return values.reshape(*leading, len(self.features)), nearest.reshape(leading), supported.reshape(leading)


    def validate(self, points, observed):
        """Score independent molecular observations against a fixed training-mean baseline.

        All finite test points are scored, including points outside the empirical
        support cutoff; coverage is reported separately. For binary targets, MSE
        is a Brier score and the baseline is the training open fraction. A skill
        score at or below zero does not improve that constant baseline.
        """
        observed = _matrix(observed, "Test observed values")
        predicted, nearest, _ = self.predict(points, max_distance=np.inf)
        if predicted.ndim != 2 or predicted.shape != observed.shape:
            raise ValueError("Test points and observed features must have corresponding rows and columns.")
        if self.binary:
            if (observed < 0).any():
                raise ValueError("Binary test observations must be nonnegative.")
            observed = (observed > 0).astype(float)
        error = np.mean((predicted-observed)**2, axis=0)
        baseline = np.mean((self.observed.mean(0)-observed)**2, axis=0)
        skill = np.divide(error, baseline, out=np.full_like(error, np.nan), where=baseline>1e-12)
        return pd.DataFrame({"feature": self.features, "test_cells": len(observed),
                             "RMSE": np.sqrt(error), "baseline_RMSE": np.sqrt(baseline),
                             "Brier" if self.binary else "MSE": error,
                             "skill_vs_training_mean": 1-skill,
                             "support_coverage": np.mean(nearest <= self.support_cutoff)})


def fit_feature_decoder(reference, values=None, *, features=None, use_rep="auto",
                        feature_key=None, layer=None, n_neighbors=30, binary=False,
                        units="Observed feature units", max_distance=None,
                        support_quantile=.99, seed=0):
    """Fit local readouts from AnnData, or from aligned latent/features arrays.

    AnnData uses its selected representation and named columns from X or layer.
    Arrays require values and feature names in matching rows. To handle large
    on-disk atlases, read_h5ad_features reads only requested molecular features.
    No scaling, time interpolation, feature discovery or peak-gene linking is
    performed implicitly. Fit RNA and ATAC decoders in their own model spaces.
    """
    if not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if hasattr(reference, "obs") and hasattr(reference, "obsm"):
        from ..anndata_io import resolve_representation
        if values is not None:
            raise ValueError("AnnData supplies observed features; do not also pass values.")
        if not reference.obs_names.is_unique:
            raise ValueError("Reference barcodes must be unique.")
        names = _names(features)
        cols = _feature_indices(reference.var, names, feature_key)
        source = reference.X if layer is None else reference.layers[layer]
        values = source[:, cols]
        barcodes = reference.obs_names.to_numpy(dtype=str)
        rep = resolve_representation(reference, use_rep)
        reference = reference.X if rep is None else reference.obsm[rep]
    else:
        names = _names(features)
        barcodes = np.arange(len(reference)).astype(str)
    reference, values = _matrix(reference, "Reference coordinates"), _matrix(values, "Observed values")
    if len(reference) != len(values) or values.shape[1] != len(names):
        raise ValueError("Reference rows, observed rows and feature names must align.")
    if binary:
        if (values < 0).any():
            raise ValueError("Peak presence/absence requires nonnegative observations.")
        values = (values > 0).astype(float)
    cutoff = _support_cutoff(reference, support_quantile, seed) if max_distance is None else float(max_distance)
    if np.isnan(cutoff) or cutoff < 0:
        raise ValueError("max_distance must be nonnegative.")
    model = NearestNeighbors(n_neighbors=min(n_neighbors, len(reference))).fit(reference)
    return FeatureDecoder(reference.copy(), values.copy(), tuple(names), model, cutoff,
                          str(units), binary, barcodes.copy())


@dataclass
class LineageSelection:
    """An annotation-conditioned model cohort, not experimental lineage tracing."""
    mask: np.ndarray
    assignments: pd.DataFrame
    targets: tuple
    support_cutoff: float

    @property
    def indices(self):
        return np.flatnonzero(self.mask)


def select_lineage(prediction, reference, *, terminal_celltypes, use_rep="auto",
                   time_key="time", celltype_key="celltype", modality="primary",
                   n_neighbors=15, min_probability=.6, max_distance=None,
                   support_quantile=.99, seed=0):
    """Select paths whose endpoints match target annotations at the terminal time.

    Uses only observed cells at prediction['times'][-1]. Missing labels become
    'unknown' and do not count as a target. Returns every path's target vote,
    nearest distance and support flag, even when no paths pass. Do not replace
    an empty selection with all paths or lower thresholds to manufacture a fate.
    """
    from ..anndata_io import resolve_representation
    if not 0 <= min_probability <= 1 or not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError("Choose a probability in [0, 1] and a positive neighbor count.")
    targets = _names(terminal_celltypes)
    if "unknown" in targets:
        raise ValueError("Unknown annotation cannot define a lineage target.")
    times = np.asarray(prediction["times"], dtype=float)
    points = np.asarray(prediction[modality], dtype=float)
    if times.ndim != 1 or len(times) < 2 or points.ndim != 3 or len(points) != len(times) or not points.size or not np.isfinite(points).all() or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Supply a finite time-by-paths-by-dimensions trajectory.")
    mask = np.isclose(reference.obs[time_key].to_numpy(dtype=float), times[-1])
    if mask.sum() < 2:
        raise ValueError("At least two annotated reference cells are needed at the terminal time.")
    terminal = reference[mask]
    rep = resolve_representation(terminal, use_rep)
    coords = _matrix(terminal.X if rep is None else terminal.obsm[rep], "Terminal reference")
    if coords.shape[1] != points.shape[2]:
        raise ValueError("Terminal reference and trajectory dimensions must match.")
    labels = terminal.obs[celltype_key].astype(object).fillna("unknown").to_numpy(dtype=str)
    if not set(targets) <= set(labels):
        raise ValueError("Requested terminal cell types are absent from the terminal reference.")
    cutoff = _support_cutoff(coords, support_quantile, seed) if max_distance is None else float(max_distance)
    if np.isnan(cutoff) or cutoff < 0:
        raise ValueError("max_distance must be nonnegative.")
    distances, indices = NearestNeighbors(n_neighbors=min(n_neighbors, len(coords))).fit(coords).kneighbors(points[-1])
    weights = _neighbor_weights(distances)
    categories = np.unique(labels)
    votes = np.stack([(weights * (labels[indices] == name)).sum(1) for name in categories], axis=1)
    target_probability = (weights * np.isin(labels[indices], targets)).sum(1)
    supported = distances[:, 0] <= cutoff
    selected = supported & (target_probability >= min_probability)
    table = pd.DataFrame({"path": np.arange(points.shape[1]), "assigned_celltype": categories[votes.argmax(1)],
                          "target_probability": target_probability, "nearest_distance": distances[:, 0],
                          "supported": supported, "selected": selected})
    return LineageSelection(selected, table, tuple(targets), cutoff)


@dataclass
class FeatureDynamics:
    """Local molecular readouts and their supported, optionally mass-weighted summaries."""
    times: np.ndarray
    values: np.ndarray
    nearest_distance: np.ndarray
    supported: np.ndarray
    summary: pd.DataFrame
    features: tuple
    units: str
    binary: bool
    weighting: str
    path_indices: np.ndarray


def compute_feature_dynamics(decoder, prediction, *, modality="primary",
                             selection=None, weighting="uniform", max_distance=None):
    """Read observed features along paths and summarize each time/feature.

    weighting='growth' uses normalized exp(log_weights) within the selected,
    supported cohort at each time. Uniform weighting gives each path equal
    weight. IQR describes variation across interpolated paths, not uncertainty
    of gene expression or experimental cell-population estimates.
    Binary observations yield local open probabilities; do not threshold an
    interpolated peak value at zero to claim a fraction of open cells.
    """
    if not isinstance(decoder, FeatureDecoder):
        raise TypeError("decoder must be a fitted FeatureDecoder.")
    times = np.asarray(prediction["times"], dtype=float)
    points = np.asarray(prediction[modality], dtype=float)
    if times.ndim != 1 or len(times) < 2 or points.ndim != 3 or len(points) != len(times) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Trajectory times must be finite and increasing with matching path states.")
    indices = np.arange(points.shape[1]) if selection is None else selection.indices
    if selection is not None and len(selection.mask) != points.shape[1]:
        raise ValueError("Lineage selection must correspond to the trajectory paths.")
    if not len(indices):
        raise ValueError("No paths passed lineage selection; inspect endpoint assignments and support.")
    values, nearest, supported = decoder.predict(points[:, indices], max_distance=max_distance)
    if weighting == "uniform":
        logw = np.zeros(points.shape[:2])
    elif weighting == "growth":
        if "log_weights" not in prediction:
            raise ValueError("Growth weighting requires an unbalanced trajectory with log_weights.")
        logw = np.asarray(prediction["log_weights"], dtype=float)
        if logw.shape == (*points.shape[:2], 1):
            logw = logw[..., 0]
        if logw.shape != points.shape[:2] or not np.isfinite(logw).all():
            raise ValueError("log_weights must be finite and match time-by-path dimensions.")
    else:
        raise ValueError("weighting must be 'uniform' or 'growth'.")
    rows = []
    for ti, time in enumerate(times):
        keep = supported[ti]
        count = int(keep.sum())
        if count:
            selected_logw = logw[ti, indices][keep]
            weights = np.exp(selected_logw - selected_logw.max())
            weights /= weights.sum()
            effective = float(1 / np.square(weights).sum())
        else:
            effective = 0.
        for fi, feature in enumerate(decoder.features):
            mean = q25 = q75 = np.nan
            if count:
                vals = values[ti, keep, fi]
                mean = float(weights @ vals)
                if weighting == "uniform":
                    q25, q75 = np.quantile(vals, [.25, .75])
                else:
                    positive = weights > 0
                    ordered_values, ordered_weights = vals[positive], weights[positive]
                    order = np.argsort(ordered_values)
                    cumulative = np.cumsum(ordered_weights[order]) - .5 * ordered_weights[order]
                    q25, q75 = np.interp([.25, .75], cumulative, ordered_values[order])
            rows.append(dict(time=float(time), feature=feature, mean=mean, q25=q25, q75=q75,
                             supported_paths=count, total_paths=len(indices), coverage=count/len(indices),
                             effective_paths=effective))
    return FeatureDynamics(times.copy(), values, nearest, supported, pd.DataFrame(rows),
                           decoder.features, decoder.units, decoder.binary, weighting, indices)


def plot_feature_dynamics(dynamics, *, axes=None, color="#0072b2", show_iqr=True):
    """Plot per-feature means and across-path IQR; missing support leaves gaps."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(len(dynamics.features), 1, figsize=(7, 2.5*len(dynamics.features)),
                               sharex=True, squeeze=False, layout="constrained")
    axes = np.asarray(axes, dtype=object).reshape(-1)
    if len(axes) != len(dynamics.features):
        raise ValueError("Provide one axis per feature.")
    for ax, feature in zip(axes, dynamics.features):
        sub = dynamics.summary[dynamics.summary.feature == feature]
        ax.plot(sub.time, sub["mean"], color=color, lw=2)
        if show_iqr:
            ax.fill_between(sub.time.to_numpy(), sub.q25.to_numpy(), sub.q75.to_numpy(), color=color, alpha=.18)
        ax.set(title=feature)
        if len(axes) == 1:
            ax.set_ylabel(dynamics.units)
        ax.spines[["top", "right"]].set_visible(False)
        if dynamics.binary:
            ax.set_ylim(0, 1)
    if len(axes) > 1:
        axes[0].figure.supylabel(dynamics.units)
    axes[-1].set_xlabel("Model time")
    return axes


def plot_feature_heatmap(dynamics, *, ax=None, standardize=False, cmap=None):
    """Plot feature-by-time means; optional row z-scores are visualization only."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(2.5, .5*len(dynamics.features))), layout="constrained")
    matrix = np.stack([dynamics.summary[dynamics.summary.feature == name]["mean"].to_numpy() for name in dynamics.features])
    if standardize:
        center, scale = np.nanmean(matrix, axis=1, keepdims=True), np.nanstd(matrix, axis=1, keepdims=True)
        matrix = (matrix-center) / np.maximum(scale, 1e-12)
    color_map = plt.get_cmap(cmap or ("coolwarm" if standardize else "viridis")).copy()
    color_map.set_bad("#e4e7eb")
    # pcolormesh keeps irregular model-time spacing correct.
    t = dynamics.times
    edges = np.r_[t[0] - .5*(t[1]-t[0]), .5*(t[:-1]+t[1:]), t[-1] + .5*(t[-1]-t[-2])]
    plot = ax.pcolormesh(edges, np.arange(len(matrix)+1), np.ma.masked_invalid(matrix), cmap=color_map, shading="flat")
    ax.set(yticks=np.arange(len(matrix))+.5, yticklabels=dynamics.features, xlabel="Model time", title="Local feature dynamics")
    ax.invert_yaxis()
    ax.figure.colorbar(plot, ax=ax, label="Row z-score" if standardize else dynamics.units)
    return ax
