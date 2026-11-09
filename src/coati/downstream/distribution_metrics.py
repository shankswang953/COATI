"""Distribution-level metrics for held-out trajectory predictions.

All coordinates must already be in the same evaluation space. The functions do
not normalize features, fit an embedding, or compare total population size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from geomloss import SamplesLoss
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class DistributionMetrics:
    """Three distribution scores and mass-weighting diagnostics."""

    support_recall: float
    sinkhorn_w2: float
    composition_similarity: float
    effective_particle_count: float
    max_particle_weight: float
    predicted_unknown_mass: float
    observed_unknown_fraction: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_particle_weights(
    n_particles: int,
    *,
    weights: np.ndarray | None = None,
    log_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return nonnegative particle weights that sum to one.

    Supply at most one of ``weights`` and ``log_weights``. Omitting both gives
    equal weights, as used for balanced trajectory models.
    """
    if n_particles < 1:
        raise ValueError("n_particles must be positive.")
    if weights is not None and log_weights is not None:
        raise ValueError("Supply weights or log_weights, not both.")
    if weights is None and log_weights is None:
        return np.full(n_particles, 1.0 / n_particles, dtype=np.float64)

    values = np.asarray(
        log_weights if log_weights is not None else weights, dtype=np.float64
    ).reshape(-1)
    if values.shape[0] != n_particles:
        raise ValueError(f"Expected {n_particles} particle weights, got {values.shape[0]}.")
    if not np.isfinite(values).all():
        raise ValueError("Particle weights must be finite.")
    if log_weights is not None:
        values = np.exp(values - values.max())
    elif np.any(values < 0):
        raise ValueError("Particle weights must be nonnegative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Particle weights must have positive total mass.")
    return values / total


def support_recall(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    log_weights: np.ndarray | None = None,
    k: int = 15,
    batch_size: int = 256,
) -> float:
    """Measure mass-aware coverage of the observed target support.

    Each reference cell receives a local radius from its k-th real-data
    neighbor. Predicted mass inside that radius is converted to an expected hit
    probability using the effective number of predicted particles.
    """
    prediction, reference = _validate_pair(prediction, reference)
    if k < 1:
        raise ValueError("k must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    particle_weights = normalize_particle_weights(
        len(prediction), weights=weights, log_weights=log_weights
    )
    effective_n = float(1.0 / np.sum(np.square(particle_weights)))
    distances = (
        NearestNeighbors(n_neighbors=min(k + 1, len(reference)), n_jobs=-1)
        .fit(reference)
        .kneighbors(reference, return_distance=True)[0]
    )
    radii = np.asarray(distances[:, -1], dtype=np.float64)
    prediction_norm = np.sum(np.square(prediction), axis=1)
    local_mass = np.empty(len(reference), dtype=np.float64)
    for start in range(0, len(reference), batch_size):
        stop = min(start + batch_size, len(reference))
        batch = reference[start:stop]
        squared_distances = (
            np.sum(np.square(batch), axis=1)[:, None]
            + prediction_norm[None, :]
            - 2.0 * (batch @ prediction.T)
        )
        membership = squared_distances <= np.square(radii[start:stop, None]) + 1e-7
        local_mass[start:stop] = membership @ particle_weights
    clipped = np.clip(local_mass, 0.0, 1.0 - 1e-15)
    hit_probability = -np.expm1(effective_n * np.log1p(-clipped))
    return float(hit_probability.mean())


def sinkhorn_w2(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    log_weights: np.ndarray | None = None,
    max_clusters: int = 512,
    blur: float = 0.05,
    random_state: int = 0,
) -> float:
    """Return ``sqrt(2 * S_epsilon)`` after weighted microclustering.

    ``S_epsilon`` is the debiased p=2 Sinkhorn divergence. This reproduces the
    W2-like score used in the gastrulation LOSO benchmark rather than exact EMD.
    """
    prediction, reference = _validate_pair(prediction, reference)
    if max_clusters < 1:
        raise ValueError("max_clusters must be positive.")
    if blur <= 0:
        raise ValueError("blur must be positive.")
    particle_weights = normalize_particle_weights(
        len(prediction), weights=weights, log_weights=log_weights
    )
    observed_weights = np.full(len(reference), 1.0 / len(reference), dtype=np.float64)
    if np.array_equal(prediction, reference) and np.allclose(
        particle_weights, observed_weights, rtol=0.0, atol=1e-12
    ):
        return 0.0
    pred_weights, pred_centers = _microcluster(
        prediction,
        particle_weights,
        min(max_clusters, len(prediction)),
        random_state,
    )
    ref_weights, ref_centers = _microcluster(
        reference,
        observed_weights,
        min(max_clusters, len(reference)),
        random_state,
    )
    loss = SamplesLoss(
        loss="sinkhorn", p=2, blur=blur, debias=True, backend="tensorized"
    )
    with torch.no_grad():
        divergence = loss(
            torch.as_tensor(pred_weights, dtype=torch.float32),
            torch.as_tensor(pred_centers, dtype=torch.float32),
            torch.as_tensor(ref_weights, dtype=torch.float32),
            torch.as_tensor(ref_centers, dtype=torch.float32),
        )
    value = max(float(divergence.detach().cpu()), 0.0)
    return float(np.sqrt(2.0 * value))


def composition_similarity(
    prediction: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    log_weights: np.ndarray | None = None,
    unknown_label: str | None = "Unknown",
) -> tuple[float, float, float]:
    """Compare predicted and observed cell-type proportions by 1NN transfer.

    Returns ``(similarity, predicted_unknown_mass, observed_unknown_fraction)``.
    Similarity is ``1 - sqrt(JSD_base2)`` after excluding and renormalizing the
    unknown category.
    """
    prediction, reference = _validate_pair(prediction, reference)
    labels = np.asarray(reference_labels).astype(str).reshape(-1)
    if labels.shape[0] != len(reference):
        raise ValueError(
            "reference_labels must contain one label per reference cell."
        )
    particle_weights = normalize_particle_weights(
        len(prediction), weights=weights, log_weights=log_weights
    )
    indices = (
        NearestNeighbors(n_neighbors=1, n_jobs=-1)
        .fit(reference)
        .kneighbors(prediction, return_distance=False)[:, 0]
    )
    assigned = labels[indices]
    categories = np.unique(labels)
    predicted = np.asarray(
        [particle_weights[assigned == category].sum() for category in categories],
        dtype=np.float64,
    )
    observed = np.asarray(
        [np.mean(labels == category) for category in categories], dtype=np.float64
    )
    unknown = np.zeros(len(categories), dtype=bool)
    if unknown_label is not None:
        unknown = np.char.lower(categories.astype(str)) == str(unknown_label).lower()
    predicted_unknown = float(predicted[unknown].sum())
    observed_unknown = float(observed[unknown].sum())
    known = ~unknown
    if (
        not np.any(known)
        or predicted[known].sum() <= 0
        or observed[known].sum() <= 0
    ):
        raise ValueError("Known cell types must have positive predicted and observed mass.")
    jsd = _js_divergence_base2(predicted[known], observed[known])
    return float(1.0 - np.sqrt(jsd)), predicted_unknown, observed_unknown


def evaluate_distribution(
    prediction: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    log_weights: np.ndarray | None = None,
    support_k: int = 15,
    support_batch_size: int = 256,
    max_clusters: int = 512,
    sinkhorn_blur: float = 0.05,
    unknown_label: str | None = "Unknown",
    random_state: int = 0,
) -> DistributionMetrics:
    """Compute support recall, W2-like distance and composition similarity."""
    prediction, reference = _validate_pair(prediction, reference)
    normalized = normalize_particle_weights(
        len(prediction), weights=weights, log_weights=log_weights
    )
    composition, predicted_unknown, observed_unknown = composition_similarity(
        prediction,
        reference,
        reference_labels,
        weights=normalized,
        unknown_label=unknown_label,
    )
    return DistributionMetrics(
        support_recall=support_recall(
            prediction,
            reference,
            weights=normalized,
            k=support_k,
            batch_size=support_batch_size,
        ),
        sinkhorn_w2=sinkhorn_w2(
            prediction,
            reference,
            weights=normalized,
            max_clusters=max_clusters,
            blur=sinkhorn_blur,
            random_state=random_state,
        ),
        composition_similarity=composition,
        effective_particle_count=float(1.0 / np.sum(np.square(normalized))),
        max_particle_weight=float(normalized.max()),
        predicted_unknown_mass=predicted_unknown,
        observed_unknown_fraction=observed_unknown,
    )


def evaluate_paired_distributions(
    predicted_primary: np.ndarray,
    observed_primary: np.ndarray,
    primary_labels: np.ndarray,
    predicted_secondary: np.ndarray,
    observed_secondary: np.ndarray,
    secondary_labels: np.ndarray,
    **kwargs,
) -> dict[str, float]:
    """Compute the three metrics independently in primary and secondary space."""
    primary = evaluate_distribution(
        predicted_primary, observed_primary, primary_labels, **kwargs
    )
    secondary = evaluate_distribution(
        predicted_secondary, observed_secondary, secondary_labels, **kwargs
    )
    return {
        **{f"primary_{key}": value for key, value in primary.as_dict().items()},
        **{f"secondary_{key}": value for key, value in secondary.as_dict().items()},
    }


def _validate_pair(
    prediction: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.ndim != 2 or reference.ndim != 2:
        raise ValueError("prediction and reference must be two-dimensional matrices.")
    if not len(prediction) or not len(reference):
        raise ValueError("prediction and reference must be nonempty.")
    if prediction.shape[1] != reference.shape[1]:
        raise ValueError("prediction and reference must have the same feature dimension.")
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("prediction and reference must contain only finite values.")
    return prediction, reference


def _microcluster(
    points: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if n_clusters >= len(points):
        return weights.astype(np.float32), points
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=2048,
        n_init=3,
        max_iter=200,
    )
    cluster_ids = model.fit_predict(points, sample_weight=weights)
    cluster_weights = np.bincount(
        cluster_ids, weights=weights, minlength=n_clusters
    ).astype(np.float64)
    keep = cluster_weights > 0
    cluster_weights = cluster_weights[keep]
    cluster_weights /= cluster_weights.sum()
    return cluster_weights.astype(np.float32), np.asarray(
        model.cluster_centers_[keep], dtype=np.float32
    )


def _js_divergence_base2(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log2(p / midpoint), 0.0).sum()
        kl_q = np.where(q > 0, q * np.log2(q / midpoint), 0.0).sum()
    return float(0.5 * (kl_p + kl_q))
