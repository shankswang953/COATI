"""Reusable RNA-to-secondary-space mapping helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


class PlainMLP(nn.Module):
    """Plain MLP checkpoint format used by TrainT/T_plain_mlp.pt."""

    def __init__(self, d_in, d_out, hidden=128, n_layers=3, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, q, t=None):
        return self.net(q)


def load_plain_mlp(path: str, device: str = "cpu") -> tuple[PlainMLP, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = PlainMLP(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, ckpt


def apply_t(model: nn.Module, x: np.ndarray, batch_size: int = 4096, device: str = "cpu") -> np.ndarray:
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).astype(np.float32)
    out = []
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            stop = min(start + batch_size, len(flat))
            xb = torch.as_tensor(flat[start:stop], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy().astype(np.float32))
    y = np.concatenate(out, axis=0)
    return y.reshape(*shape[:-1], y.shape[-1])


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return 1.0 - num / np.maximum(den, eps)


def t_error_table(
    primary_ref: np.ndarray,
    secondary_ref: np.ndarray,
    labels: np.ndarray | None = None,
    stages: np.ndarray | None = None,
    model: nn.Module | None = None,
    t_checkpoint: str | None = None,
    k: int = 30,
    device: str = "cpu",
) -> pd.DataFrame:
    if model is None:
        if t_checkpoint is None:
            raise ValueError("Provide either model or t_checkpoint.")
        model, _ = load_plain_mlp(t_checkpoint, device=device)
    pred = apply_t(model, primary_ref, device=device)
    err = pred - secondary_ref
    df = pd.DataFrame({
        "cell_index": np.arange(len(primary_ref)),
        "t_l2_error": np.linalg.norm(err, axis=1),
        "t_cosine_distance": cosine_distance(pred, secondary_ref),
    })
    if labels is not None:
        df["cell_type"] = labels.astype(str)
    if stages is not None:
        df["stage"] = stages.astype(str)

    nn = NearestNeighbors(n_neighbors=min(k, len(secondary_ref)), algorithm="auto")
    nn.fit(secondary_ref)
    dist, idx = nn.kneighbors(pred, return_distance=True)
    df["pred_nn_distance_1"] = dist[:, 0]
    df["pred_nn_distance_mean"] = dist.mean(axis=1)
    if labels is not None:
        same_label = labels[idx].astype(str) == labels.astype(str)[:, None]
        df["pred_nn_same_cell_type_frac"] = same_label.mean(axis=1)
    if stages is not None:
        same_stage = stages[idx].astype(str) == stages.astype(str)[:, None]
        df["pred_nn_same_stage_frac"] = same_stage.mean(axis=1)
    return df


def summarize_t_error(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("t_l2_error", "size"),
            mean_l2=("t_l2_error", "mean"),
            median_l2=("t_l2_error", "median"),
            p90_l2=("t_l2_error", lambda x: x.quantile(0.90)),
            mean_cosine=("t_cosine_distance", "mean"),
            mean_pred_nn_same_cell_type=("pred_nn_same_cell_type_frac", "mean")
            if "pred_nn_same_cell_type_frac" in df.columns else ("t_l2_error", "size"),
            mean_pred_nn_same_stage=("pred_nn_same_stage_frac", "mean")
            if "pred_nn_same_stage_frac" in df.columns else ("t_l2_error", "size"),
        )
        .reset_index()
    )

