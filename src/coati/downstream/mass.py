"""Endpoint decomposition of model-predicted cell-type proportions."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .features import _matrix, _neighbor_weights
from ..anndata_io import observation_times, resolve_representation


def _time_index(times, value, name):
    value = float(value)
    matches = np.flatnonzero(np.isclose(times, value, rtol=0, atol=1e-8))
    if not np.isfinite(value) or len(matches) != 1:
        raise ValueError(f"{name} must match exactly one trajectory time; include it in the prediction grid.")
    return int(matches[0])


def _assign(points, coordinates, labels, n_neighbors):
    neighbors = NearestNeighbors(n_neighbors=min(n_neighbors, len(coordinates))).fit(coordinates)
    categories = np.unique(labels)
    assigned, nearest, confidence = [], [], []
    for block in np.array_split(points, max(1, int(np.ceil(len(points) / 1024)))):
        distances, indices = neighbors.kneighbors(block)
        weights = _neighbor_weights(distances)
        votes = np.stack([(weights * (labels[indices] == name)).sum(1) for name in categories], axis=1)
        chosen = votes.argmax(1)
        assigned.append(categories[chosen])
        nearest.append(distances[:, 0])
        confidence.append(votes[np.arange(len(block)), chosen])
    return np.concatenate(assigned), np.concatenate(nearest), np.concatenate(confidence)


@dataclass
class MassDecomposition:
    """Relative mass components, endpoint assignments and transition summaries.

    Incoming and outgoing are nonnegative fractions; net_change equals
    incoming - outgoing + reweighting. assignments retains every particle,
    including particles outside the requested type. No paths are discarded.
    """
    summary: pd.DataFrame
    assignments: pd.DataFrame
    transitions: pd.DataFrame
    metadata: dict

    def plot(self, *, ax=None, show_observed=False, title=None):
        """Plot signed components in percentage points; optionally show atlas change."""
        return plot_mass_decomposition(self, ax=ax, show_observed=show_observed, title=title)


def decompose_mass(prediction, reference, *, celltype, start_time=None, end_time=None,
                   use_rep="auto", time_key="time", celltype_key="celltype",
                   time_mapping=None, n_neighbors=1):
    """Separate a cell-type proportion change into incoming, outgoing and reweighting.

    Track the same ordered particles at two observed times. prediction must be
    in the same feature units as reference (Result.predict_trajectory(raw=True)).
    Labels are transferred independently from stage-matched reference cells by
    hard distance-weighted kNN in the primary training representation, not UMAP.
    Missing annotations become 'unknown'. One neighbor matches the paper figure;
    larger counts are an explicit analysis choice. Ties use sorted label order.

    Normalize exp(log_weights) over the FULL supplied population at each time.
    Incoming uses later weights, outgoing uses earlier weights, and reweighting
    uses weight differences for retained particles. Do not preselect a lineage
    or a cell type. Missing log_weights means uniform balanced weights.

    Endpoints default to the first/last prediction times and must match observed
    reference stages. The observed_change column is a cross-sectional atlas
    proportion difference, not a measured transition of the predicted cohort.
    Relative reweighting is not an absolute proliferation or death estimate.
    """
    if not isinstance(celltype, str) or not celltype or celltype == "unknown":
        raise ValueError("celltype must name a known annotated cell type.")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    times = np.asarray(prediction["times"], dtype=float)
    points = np.asarray(prediction["primary"], dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or points.ndim != 3
            or points.shape[0] != len(times) or not min(points.shape)
            or not np.isfinite(points).all()):
        raise ValueError("Supply a finite, increasing time-by-particles-by-dimensions trajectory.")
    first = _time_index(times, times[0] if start_time is None else start_time, "start_time")
    last = _time_index(times, times[-1] if end_time is None else end_time, "end_time")
    if first >= last:
        raise ValueError("start_time must precede end_time.")
    if celltype_key not in reference.obs:
        raise KeyError(f"Missing reference.obs[{celltype_key!r}].")
    reference_times = observation_times(reference, time_key=time_key, time_mapping=time_mapping)
    rep = resolve_representation(reference, use_rep)
    coordinates = _matrix(reference.X if rep is None else reference.obsm[rep], "Reference")
    if coordinates.shape != (reference.n_obs, points.shape[2]):
        raise ValueError("Reference and trajectory dimensions must match in the same training representation.")
    labels = reference.obs[celltype_key].astype(object).fillna("unknown").to_numpy(dtype=str)
    stage_masks = [np.isclose(reference_times, times[i], rtol=0, atol=1e-8) for i in (first, last)]
    if any(not mask.any() for mask in stage_masks):
        raise ValueError("Both endpoints need an observed reference stage; no nearest-stage substitution is made.")
    if not any(celltype in labels[mask] for mask in stage_masks):
        raise ValueError(f"Cell type {celltype!r} is absent from both endpoint references.")
    if "log_weights" in prediction:
        logw = np.asarray(prediction["log_weights"], dtype=float)
        if logw.shape == (*points.shape[:2], 1):
            logw = logw[..., 0]
        if logw.shape != points.shape[:2] or not np.isfinite(logw).all():
            raise ValueError("log_weights must be finite and match the full time-by-particle axes.")
        weights = np.exp(logw[[first, last]] - logw[[first, last]].max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        weighting = "normalized predicted mass"
    else:
        weights = np.full((2, points.shape[1]), 1 / points.shape[1])
        weighting = "uniform balanced mass"
    endpoints = [_assign(points[i], coordinates[mask], labels[mask], n_neighbors)
                 for i, mask in zip((first, last), stage_masks)]
    label0, distance0, confidence0 = endpoints[0]
    label1, distance1, confidence1 = endpoints[1]
    is0, is1 = label0 == celltype, label1 == celltype
    retained, incoming, outgoing = is0 & is1, ~is0 & is1, is0 & ~is1
    w0, w1 = weights
    start, end = float(w0[is0].sum()), float(w1[is1].sum())
    inc, out = float(w1[incoming].sum()), float(w0[outgoing].sum())
    rew = float((w1[retained] - w0[retained]).sum())
    observed = [float((labels[mask] == celltype).mean()) for mask in stage_masks]
    summary = pd.DataFrame([dict(celltype=celltype, start_time=times[first], end_time=times[last],
        start_proportion=start, end_proportion=end, incoming=inc, outgoing=out,
        reweighting=rew, net_change=end-start, closure_error=(inc-out+rew)-(end-start),
        observed_start_proportion=observed[0], observed_end_proportion=observed[1],
        observed_change=observed[1]-observed[0], particles=points.shape[1])]).set_index("celltype")
    assignments = pd.DataFrame(dict(particle=np.arange(points.shape[1]),
        start_celltype=label0, end_celltype=label1, start_weight=w0, end_weight=w1,
        component=np.select([retained, incoming, outgoing], ["reweighting", "incoming", "outgoing"], default="other"),
        start_distance=distance0, end_distance=distance1,
        start_vote=confidence0, end_vote=confidence1))
    transitions = assignments.groupby(["start_celltype", "end_celltype"], observed=True).agg(
        particles=("particle", "size"), start_mass=("start_weight", "sum"), end_mass=("end_weight", "sum")).reset_index()
    metadata = dict(weighting=weighting, use_rep="X" if rep is None else rep,
        time_key=time_key, celltype_key=celltype_key, n_neighbors=int(n_neighbors),
        effective_neighbors=[min(int(n_neighbors), int(mask.sum())) for mask in stage_masks],
        reference_cells=[int(mask.sum()) for mask in stage_masks],
        unknown_reference_cells=[int((labels[mask] == "unknown").sum()) for mask in stage_masks],
        classification="stage-matched hard distance-weighted kNN", population="all supplied ordered particles")
    return MassDecomposition(summary, assignments, transitions, metadata)


def plot_mass_decomposition(decomposition, *, ax=None, show_observed=False, title=None):
    """Display incoming, negative outgoing and retained reweighting in percentage points."""
    import matplotlib.pyplot as plt
    if not isinstance(decomposition, MassDecomposition):
        raise TypeError("decomposition must be a MassDecomposition.")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.1))
    row = decomposition.summary.iloc[0]
    positive = negative = 0.
    for label, value, color in [("Reweighting", row.reweighting, "#3274AD"),
                                 ("Outgoing", -row.outgoing, "#CC79A7"),
                                 ("Incoming", row.incoming, "#DFA12D")]:
        value *= 100
        left = positive if value >= 0 else negative
        ax.barh(0, value, left=left, height=.4, color=color, label=label)
        if value >= 0:
            positive += value
        else:
            negative += value
    ax.axvline(0, color="#9ba5af", linewidth=.9)
    ax.scatter([100 * row.net_change], [0], marker="D", s=42, color="#263746", zorder=4, label="Predicted net")
    if show_observed:
        ax.scatter([100 * row.observed_change], [.30], marker="D", s=44,
                   facecolor="none", edgecolor="#D85261", linewidth=1.6, label="Observed net", zorder=4)
    ax.set(yticks=[], ylim=(-.45, .5), xlabel="Change in cell-type proportion (percentage points)",
           title=title or f"{decomposition.summary.index[0]} · t = {row.start_time:g} → {row.end_time:g}")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.margins(x=.2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.42), ncol=3, frameon=False, fontsize=9)
    ax.figure.subplots_adjust(bottom=.38, top=.80)
    return ax
