"""kNN label transfer and trajectory-distance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def fit_knn(reference: np.ndarray, k: int) -> NearestNeighbors:
    nn = NearestNeighbors(n_neighbors=min(k, len(reference)), algorithm="auto")
    nn.fit(reference)
    return nn


def knn_indices_for_trajectory(traj: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    nn = fit_knn(reference, k)
    idx_by_t = []
    for t in range(traj.shape[0]):
        _, idx = nn.kneighbors(traj[t], return_distance=True)
        idx_by_t.append(idx.astype(np.int32))
    return np.stack(idx_by_t, axis=0)


def label_distribution_from_indices(
    idx: np.ndarray,
    labels: np.ndarray,
    categories: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if categories is None:
        categories = sorted(np.unique(labels.astype(str)).tolist())
    cat_to_i = {c: i for i, c in enumerate(categories)}
    votes = labels[idx].astype(str)
    probs = np.zeros((*idx.shape[:-1], len(categories)), dtype=np.float32)
    flat_votes = votes.reshape(-1, votes.shape[-1])
    flat_probs = probs.reshape(-1, probs.shape[-1])
    for i, row in enumerate(flat_votes):
        vals, counts = np.unique(row, return_counts=True)
        for val, count in zip(vals, counts):
            if val in cat_to_i:
                flat_probs[i, cat_to_i[val]] = count / row.shape[0]
    return probs, categories


def majority_from_probs(probs: np.ndarray, categories: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cats = np.asarray(categories, dtype=object)
    best = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    return cats[best], conf.astype(np.float32)


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    n_cat = probs.shape[-1]
    if n_cat <= 1:
        return np.zeros(probs.shape[:-1], dtype=np.float32)
    h = -(probs * np.log2(probs + 1e-12)).sum(axis=-1) / np.log2(n_cat)
    return h.astype(np.float32)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), 1e-12)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log2(p + 1e-12) - np.log2(m + 1e-12))).sum(axis=-1)
    kl_qm = (q * (np.log2(q + 1e-12) - np.log2(m + 1e-12))).sum(axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def js_similarity(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (1.0 - js_divergence(p, q)).astype(np.float32)


def total_variation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (0.5 * np.abs(p - q).sum(axis=-1)).astype(np.float32)


def zscore_features(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def chamfer_and_centroid(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    coords: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    n = idx_a.shape[0]
    chamfer = np.empty(n, dtype=np.float32)
    centroid = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        a = coords[idx_a[start:stop]]
        b = coords[idx_b[start:stop]]
        ca = a.mean(axis=1)
        cb = b.mean(axis=1)
        centroid[start:stop] = np.linalg.norm(ca - cb, axis=1)
        diff = a[:, :, None, :] - b[:, None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        chamfer[start:stop] = 0.5 * (
            dist.min(axis=2).mean(axis=1) + dist.min(axis=1).mean(axis=1)
        )
    return chamfer, centroid


def trajectory_chamfer_summary(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    coords: np.ndarray,
    prefix: str = "",
    batch_size: int = 256,
) -> pd.DataFrame:
    t_steps, n_cells, _ = idx_a.shape
    chamfer_by_t = np.empty((t_steps, n_cells), dtype=np.float32)
    centroid_by_t = np.empty((t_steps, n_cells), dtype=np.float32)
    for t in range(t_steps):
        chamfer, centroid = chamfer_and_centroid(idx_a[t], idx_b[t], coords, batch_size=batch_size)
        chamfer_by_t[t] = chamfer
        centroid_by_t[t] = centroid
    p = f"{prefix}_" if prefix else ""
    return pd.DataFrame({
        f"{p}terminal_chamfer": chamfer_by_t[-1],
        f"{p}trajectory_chamfer_mean": chamfer_by_t.mean(axis=0),
        f"{p}trajectory_chamfer_mean_no_terminal": chamfer_by_t[:-1].mean(axis=0),
        f"{p}terminal_centroid": centroid_by_t[-1],
        f"{p}trajectory_centroid_mean": centroid_by_t.mean(axis=0),
        f"{p}trajectory_centroid_mean_no_terminal": centroid_by_t[:-1].mean(axis=0),
    })

