"""Experiment-agnostic downstream analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .knn_metrics import (
    entropy_from_probs,
    js_similarity,
    knn_indices_for_trajectory,
    label_distribution_from_indices,
    majority_from_probs,
    total_variation,
    trajectory_chamfer_summary,
    zscore_features,
)


def same_space_terminal_trajectory_difference(
    reference_traj: np.ndarray,
    query_traj: np.ndarray,
    reference_points: np.ndarray,
    labels: np.ndarray | None = None,
    initial_labels: np.ndarray | None = None,
    k: int = 30,
    batch_size: int = 256,
) -> pd.DataFrame:
    ref_knn = knn_indices_for_trajectory(reference_traj, reference_points, k)
    query_knn = knn_indices_for_trajectory(query_traj, reference_points, k)
    df = trajectory_chamfer_summary(ref_knn, query_knn, reference_points, batch_size=batch_size)
    df.insert(0, "cell_index", np.arange(reference_traj.shape[1]))
    if initial_labels is not None:
        df["initial_cell_type"] = initial_labels.astype(str)
    if labels is not None:
        probs_ref, cats = label_distribution_from_indices(ref_knn[-1], labels)
        probs_query, _ = label_distribution_from_indices(query_knn[-1], labels, cats)
        hard_ref, conf_ref = majority_from_probs(probs_ref, cats)
        hard_query, conf_query = majority_from_probs(probs_query, cats)
        df["reference_terminal_majority_cell_type"] = hard_ref
        df["query_terminal_majority_cell_type"] = hard_query
        df["terminal_cell_type_disagree"] = hard_ref != hard_query
        df["reference_terminal_confidence"] = conf_ref
        df["query_terminal_confidence"] = conf_query
    return df


def paired_modality_terminal_trajectory_difference(
    primary_traj: np.ndarray,
    secondary_traj: np.ndarray,
    primary_ref: np.ndarray,
    secondary_ref: np.ndarray,
    labels: np.ndarray | None = None,
    initial_labels: np.ndarray | None = None,
    k: int = 30,
    batch_size: int = 256,
) -> pd.DataFrame:
    primary_knn = knn_indices_for_trajectory(primary_traj, primary_ref, k)
    secondary_knn = knn_indices_for_trajectory(secondary_traj, secondary_ref, k)
    joint_ref = np.concatenate([zscore_features(primary_ref), zscore_features(secondary_ref)], axis=1)
    metrics = [
        trajectory_chamfer_summary(primary_knn, secondary_knn, primary_ref, "primary_space", batch_size),
        trajectory_chamfer_summary(primary_knn, secondary_knn, secondary_ref, "secondary_space", batch_size),
        trajectory_chamfer_summary(primary_knn, secondary_knn, joint_ref, "joint_space", batch_size),
    ]
    df = pd.DataFrame({"cell_index": np.arange(primary_traj.shape[1])})
    if initial_labels is not None:
        df["initial_cell_type"] = initial_labels.astype(str)
    if labels is not None:
        p_prob, cats = label_distribution_from_indices(primary_knn[-1], labels)
        s_prob, _ = label_distribution_from_indices(secondary_knn[-1], labels, cats)
        p_hard, p_conf = majority_from_probs(p_prob, cats)
        s_hard, s_conf = majority_from_probs(s_prob, cats)
        df["primary_terminal_majority_cell_type"] = p_hard
        df["secondary_terminal_majority_cell_type"] = s_hard
        df["terminal_cell_type_disagree"] = p_hard != s_hard
        df["primary_terminal_confidence"] = p_conf
        df["secondary_terminal_confidence"] = s_conf
    return pd.concat([df] + metrics, axis=1)


def rna_atac_disagreement_by_time(
    primary_traj: np.ndarray,
    secondary_traj: np.ndarray,
    primary_ref: np.ndarray,
    secondary_ref: np.ndarray,
    labels: np.ndarray,
    initial_labels: np.ndarray | None = None,
    time_indices: list[int] | None = None,
    time_names: list[str] | None = None,
    k: int = 30,
) -> pd.DataFrame:
    if time_indices is None:
        time_indices = [primary_traj.shape[0] // 2, primary_traj.shape[0] - 1]
    if time_names is None:
        time_names = [f"time_{i}" for i in time_indices]
    p_knn = knn_indices_for_trajectory(primary_traj, primary_ref, k)
    s_knn = knn_indices_for_trajectory(secondary_traj, secondary_ref, k)
    categories = sorted(np.unique(labels.astype(str)).tolist())
    rows = []
    for tidx, tname in zip(time_indices, time_names):
        p_prob, _ = label_distribution_from_indices(p_knn[tidx], labels, categories)
        s_prob, _ = label_distribution_from_indices(s_knn[tidx], labels, categories)
        p_hard, p_conf = majority_from_probs(p_prob, categories)
        s_hard, s_conf = majority_from_probs(s_prob, categories)
        tv = total_variation(p_prob, s_prob)
        js = js_similarity(p_prob, s_prob)
        out = pd.DataFrame({
            "cell_index": np.arange(primary_traj.shape[1]),
            "time_index": tidx,
            "time_name": tname,
            "primary_top": p_hard,
            "primary_top_prob": p_conf,
            "secondary_top": s_hard,
            "secondary_top_prob": s_conf,
            "top_label_disagree": p_hard != s_hard,
            "tv_distance": tv,
            "js_similarity": js,
        })
        if initial_labels is not None:
            out["initial_cell_type"] = initial_labels.astype(str)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def time_resolved_cross_modal_consistency(
    primary_traj: np.ndarray,
    secondary_traj: np.ndarray,
    primary_ref: np.ndarray,
    secondary_ref: np.ndarray,
    labels: np.ndarray,
    categories: list[str] | None = None,
    k: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p_knn = knn_indices_for_trajectory(primary_traj, primary_ref, k)
    s_knn = knn_indices_for_trajectory(secondary_traj, secondary_ref, k)
    categories = categories or sorted(np.unique(labels.astype(str)).tolist())
    rows = []
    per_cell_rows = []
    for t in range(primary_traj.shape[0]):
        p_prob, _ = label_distribution_from_indices(p_knn[t], labels, categories)
        s_prob, _ = label_distribution_from_indices(s_knn[t], labels, categories)
        p_hard, p_conf = majority_from_probs(p_prob, categories)
        s_hard, s_conf = majority_from_probs(s_prob, categories)
        sim = js_similarity(p_prob, s_prob)
        match = p_hard == s_hard
        p_ent = entropy_from_probs(p_prob)
        s_ent = entropy_from_probs(s_prob)
        rows.append({
            "time_index": t,
            "mean_js_similarity": float(np.mean(sim)),
            "median_js_similarity": float(np.median(sim)),
            "label_match_rate": float(np.mean(match)),
            "mean_primary_confidence": float(np.mean(p_conf)),
            "mean_secondary_confidence": float(np.mean(s_conf)),
            "mean_primary_entropy": float(np.mean(p_ent)),
            "mean_secondary_entropy": float(np.mean(s_ent)),
        })
        per_cell_rows.append(pd.DataFrame({
            "time_index": t,
            "cell_index": np.arange(primary_traj.shape[1]),
            "js_similarity": sim,
            "primary_label": p_hard,
            "secondary_label": s_hard,
            "label_match": match,
            "primary_confidence": p_conf,
            "secondary_confidence": s_conf,
            "primary_entropy": p_ent,
            "secondary_entropy": s_ent,
        }))
    return pd.DataFrame(rows), pd.concat(per_cell_rows, ignore_index=True)

