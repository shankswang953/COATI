"""Reusable negative-control helpers for gene / peak support analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_quantile_bins(df: pd.DataFrame, columns: list[str], n_bins: int = 5) -> pd.DataFrame:
    """Add quantile-bin columns for matching controls by continuous features."""
    out = df.copy()
    for col in columns:
        valid = out[col].notna()
        labels = pd.Series(np.nan, index=out.index, dtype=object)
        if valid.sum() > 1:
            labels.loc[valid] = pd.qcut(
                out.loc[valid, col],
                q=min(n_bins, valid.sum()),
                duplicates="drop",
                labels=False,
            ).astype(str)
        out[f"{col}_bin"] = labels
    return out


def matched_random_controls(
    feature_table: pd.DataFrame,
    target_genes: list[str],
    gene_col: str = "gene",
    match_cols: list[str] | None = None,
    n_controls_per_gene: int = 20,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample genes matched by precomputed categorical / binned columns.

    `feature_table` should contain one row per gene. `match_cols` can include
    expression bins, peak-count bins, GC bins, etc. The function avoids sampling
    target genes as controls for another target gene.
    """
    match_cols = match_cols or []
    rng = np.random.default_rng(random_state)
    table = feature_table.copy()
    targets = set(map(str, target_genes))
    rows = []
    for target in target_genes:
        target = str(target)
        target_rows = table[table[gene_col].astype(str).eq(target)]
        if target_rows.empty:
            rows.append({
                "target_gene": target,
                "control_gene": None,
                "status": "target_missing",
            })
            continue
        target_row = target_rows.iloc[0]
        pool = table[~table[gene_col].astype(str).isin(targets)].copy()
        for col in match_cols:
            pool = pool[pool[col].astype(str).eq(str(target_row[col]))]
        if pool.empty:
            rows.append({
                "target_gene": target,
                "control_gene": None,
                "status": "no_matched_pool",
            })
            continue
        take = min(n_controls_per_gene, len(pool))
        chosen_idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
        for idx in chosen_idx:
            rows.append({
                "target_gene": target,
                "control_gene": str(table.loc[idx, gene_col]),
                "status": "matched",
            })
    return pd.DataFrame(rows)


def summarize_target_vs_controls(
    values: pd.DataFrame,
    controls: pd.DataFrame,
    gene_col: str = "gene",
    value_col: str = "score",
) -> pd.DataFrame:
    """Compare target-gene scores with matched-control score distributions."""
    value_map = values.set_index(gene_col)[value_col]
    rows = []
    for target, sub in controls.groupby("target_gene", dropna=False):
        target_value = value_map.get(target, np.nan)
        control_values = value_map.reindex(sub["control_gene"].dropna().astype(str)).dropna()
        if control_values.empty or pd.isna(target_value):
            rows.append({
                "target_gene": target,
                "target_score": target_value,
                "n_controls": int(len(control_values)),
                "control_mean": np.nan,
                "control_sd": np.nan,
                "empirical_p_high": np.nan,
                "z_vs_controls": np.nan,
            })
            continue
        rows.append({
            "target_gene": target,
            "target_score": float(target_value),
            "n_controls": int(len(control_values)),
            "control_mean": float(control_values.mean()),
            "control_sd": float(control_values.std(ddof=1)),
            "empirical_p_high": float((1 + (control_values >= target_value).sum()) / (len(control_values) + 1)),
            "z_vs_controls": float((target_value - control_values.mean()) / (control_values.std(ddof=1) + 1e-8)),
        })
    return pd.DataFrame(rows)
