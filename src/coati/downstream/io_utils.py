"""I/O helpers shared by downstream analyses."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch


DEFAULT_TIME_KEYS = ("time_0", "time_1", "time_2")


@dataclass(frozen=True)
class TimeReference:
    x: np.ndarray
    labels: np.ndarray | None = None
    stage: np.ndarray | None = None
    offsets: np.ndarray | None = None


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_scale(norm_path: str | None) -> float:
    if norm_path is None:
        return 1.0
    obj = torch.load(norm_path, map_location="cpu")
    if isinstance(obj, dict) and "scale" in obj:
        return float(obj["scale"])
    if isinstance(obj, (float, int)):
        return float(obj)
    raise ValueError(f"Cannot read scale from {norm_path}")


def load_npz_by_time(
    path: str,
    norm_path: str | None = None,
    time_keys: Iterable[str] = DEFAULT_TIME_KEYS,
    dtype=np.float32,
) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    scale = load_scale(norm_path)
    out = {}
    for key in time_keys:
        if key not in z:
            raise KeyError(f"Missing {key} in {path}")
        out[key] = np.asarray(z[key], dtype=dtype) / scale
    return out


def load_labels_by_time(
    path: str,
    time_keys: Iterable[str] = DEFAULT_TIME_KEYS,
) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {key: z[key].astype(str) for key in time_keys}


def concat_by_time(
    data: dict[str, np.ndarray],
    labels: dict[str, np.ndarray] | None = None,
    time_keys: Iterable[str] = DEFAULT_TIME_KEYS,
    stage_names: Iterable[str] | None = None,
) -> TimeReference:
    keys = list(time_keys)
    stages = list(stage_names) if stage_names is not None else keys
    arrays = [data[k] for k in keys]
    offsets = np.cumsum([0] + [len(x) for x in arrays])
    stage = np.concatenate([
        np.full(len(data[k]), stages[i], dtype=object)
        for i, k in enumerate(keys)
    ])
    concat_labels = None
    if labels is not None:
        concat_labels = np.concatenate([labels[k].astype(str) for k in keys])
    return TimeReference(
        x=np.concatenate(arrays, axis=0),
        labels=concat_labels,
        stage=stage,
        offsets=offsets,
    )


def load_trajectory(path: str) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().numpy()
    return np.asarray(obj, dtype=np.float32)


def save_table(df: pd.DataFrame, path: str) -> str:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    return path


def parse_weight_from_name(path: str, key: str = "a") -> float:
    name = os.path.basename(path)
    match = re.search(rf"_{re.escape(key)}([0-9.]+)\.pt$", name)
    if not match:
        raise ValueError(f"Cannot parse _{key}<value>.pt from {path}")
    return float(match.group(1))


def discover_weighted_trajectories(
    directory: str,
    prefix: str,
    key: str = "a",
    include_zero: bool = True,
) -> list[tuple[float, str]]:
    pairs = []
    for name in os.listdir(directory):
        if not (name.startswith(prefix) and name.endswith(".pt")):
            continue
        path = os.path.join(directory, name)
        try:
            w = parse_weight_from_name(path, key=key)
        except ValueError:
            continue
        if include_zero or w != 0:
            pairs.append((w, path))
    return sorted(pairs, key=lambda x: x[0])

