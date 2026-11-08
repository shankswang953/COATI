"""Exact ToySplit data generation and visualization for the BranchSync tutorial."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn


TIME_LABELS = [f"time_{index}" for index in range(5)]
TIME_POINTS = np.arange(5, dtype=float)
TIME_EDGES = np.linspace(0.0, 1.0, 6)


class HistoricalFiLMMap(nn.Module):
    """FiLM map used by the original ToySplit experiment."""

    def __init__(self, d_in=2, d_out=2, hidden=64, n_layers=3,
                 t_hidden=32, dropout=0.0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.hidden = hidden
        self.n_layers = n_layers
        self.t_hidden = t_hidden
        self.dropout_p = dropout
        self.n_modulated = max(n_layers - 1, 1)
        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_hidden),
            nn.GELU(),
            nn.Linear(t_hidden, 2 * hidden * self.n_modulated),
        )
        nn.init.zeros_(self.t_encoder[-1].weight)
        nn.init.zeros_(self.t_encoder[-1].bias)
        self.in_proj = nn.Linear(d_in, hidden)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden, hidden) for _ in range(self.n_modulated - 1)
        )
        self.out_proj = nn.Linear(hidden, d_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q, t):
        t = torch.as_tensor(t, dtype=q.dtype, device=q.device)
        t = torch.broadcast_to(t, q.shape[:-1]).unsqueeze(-1)
        film = self.t_encoder(t).reshape(
            *t.shape[:-1], self.n_modulated, 2, self.hidden
        )
        h = self.in_proj(q)
        for index, layer in enumerate([nn.Identity(), *self.hidden_layers]):
            if index:
                h = layer(h)
            h = h * (1.0 + film[..., index, 0, :]) + film[..., index, 1, :]
            h = self.dropout(self.activation(h))
        return self.out_proj(h)


def _map_stage_metrics(model, primary, secondary, sinkhorn):
    rows = []
    model.eval()
    with torch.no_grad():
        for index, (x, y, loss_fn) in enumerate(zip(primary, secondary, sinkhorn)):
            predicted = model(x, float(TIME_POINTS[index]))
            mse = float((predicted - y).square().mean().detach().cpu())
            residual = (predicted - y).square().sum()
            centered = (y - y.mean(0, keepdim=True)).square().sum().clamp_min(1e-12)
            rows.append({
                "time": float(TIME_POINTS[index]),
                "validation_cells": len(x),
                "paired_rmse": math.sqrt(mse),
                "r2": float((1.0 - residual / centered).detach().cpu()),
                "sinkhorn": float(loss_fn(predicted, y).detach().cpu()),
            })
    model.train()
    return rows


def train_historical_map(cells, directory, seed=0, device="cpu"):
    """Refit T with the original raw-coordinate ToySplit recipe.

    This intentionally does not standardize coordinates or time. It uses a paired
    80/20 split within every stage, time-balanced raw MSE, and selects every fifth
    epoch by the sum of the five validation Sinkhorn divergences.
    """
    from geomloss import SamplesLoss

    device = torch.device(device)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    split_generator = torch.Generator(device="cpu").manual_seed(seed)
    train_x, train_y, val_x, val_y, saved_splits = [], [], [], [], {}
    times = cells.obs["time"].to_numpy(float)
    for index, time in enumerate(TIME_POINTS):
        mask = times == time
        x = torch.tensor(cells.obsm["X_latent"][mask], dtype=torch.float32, device=device)
        y = torch.tensor(cells.obsm["X_secondary"][mask], dtype=torch.float32, device=device)
        permutation = torch.randperm(len(x), generator=split_generator)
        n_val = max(1, int(round(len(x) * 0.20)))
        validation = permutation[:n_val]
        training = permutation[n_val:]
        train_x.append(x[training.to(device)])
        train_y.append(y[training.to(device)])
        val_x.append(x[validation.to(device)])
        val_y.append(y[validation.to(device)])
        saved_splits[f"train_{index}"] = training.numpy()
        saved_splits[f"validation_{index}"] = validation.numpy()
    np.savez(directory / "splits.npz", **saved_splits)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HistoricalFiLMMap().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    blurs = [max(0.05 * float(torch.pdist(y).median()), 1e-6) for y in
             [torch.tensor(cells.obsm["X_secondary"][times == time], dtype=torch.float32)
              for time in TIME_POINTS]]
    sinkhorn = [SamplesLoss("sinkhorn", p=2, blur=blur).to(device) for blur in blurs]

    flat_x = torch.cat(train_x)
    flat_y = torch.cat(train_y)
    flat_t = torch.cat([torch.full((len(x),), float(time), device=device)
                        for x, time in zip(train_x, TIME_POINTS)])
    flat_weight = torch.cat([torch.full((len(x),), 1.0 / len(x), device=device)
                             for x in train_x])
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    best_sum, best_epoch, best_state, best_rows = math.inf, 0, None, None
    history = []
    for epoch in range(1, 301):
        model.train()
        for rows in torch.randperm(len(flat_x), generator=batch_generator).split(256):
            rows = rows.to(device)
            prediction = model(flat_x[rows], flat_t[rows])
            per_cell_mse = (prediction - flat_y[rows]).square().mean(1)
            loss = (per_cell_mse * flat_weight[rows]).sum() / flat_weight[rows].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            stage_rows = _map_stage_metrics(model, val_x, val_y, sinkhorn)
            validation_sum = sum(row["sinkhorn"] for row in stage_rows)
            history.append({"epoch": epoch, "validation_sinkhorn_sum": validation_sum})
            if validation_sum < best_sum:
                best_sum = validation_sum
                best_epoch = epoch
                best_rows = stage_rows
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    report = {
        "status": "completed",
        "selection": "minimum sum of per-stage validation Sinkhorn divergences",
        "coordinate_space": "raw",
        "normalization": None,
        "best_epoch": best_epoch,
        "best_validation_sinkhorn_sum": best_sum,
        "secondary_blurs": blurs,
        "stages": best_rows,
    }
    torch.save({"state_dict": best_state, "config": {
        "d_in": 2, "d_out": 2, "hidden": 64, "n_layers": 3,
        "t_hidden": 32, "dropout": 0.0,
    }, "meta": report}, directory / "T_FiLM.pt")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, report


def generate_branch_sync(n_per_time=400, seed=42):
    """Reproduce the paired ToySplit observations used for Figure 2b."""
    if n_per_time % 2:
        raise ValueError("n_per_time must be even so both branches are balanced.")
    rng = np.random.default_rng(seed)
    primary, secondary, branches = {}, {}, {}
    for time_index, label in enumerate(TIME_LABELS):
        left, right = TIME_EDGES[time_index:time_index + 2]
        step = (right - left) / n_per_time
        continuous_time = np.linspace(left, right, n_per_time, endpoint=False) + 0.5 * step
        upper = np.arange(n_per_time) % 2 == 0
        if time_index == 0:
            primary_y = rng.normal(0.50, 0.025, n_per_time)
            secondary_y = primary_y.copy()
            branch = np.full(n_per_time, "shared", dtype=object)
        elif time_index == 1:
            means = np.where(upper, 0.52, 0.48)
            primary_y = rng.normal(means, 0.025)
            displacement = np.where(
                upper,
                np.exp(primary_y - 0.51) * continuous_time * 0.1,
                -np.exp(0.49 - primary_y) * continuous_time * 0.1,
            )
            secondary_y = primary_y + displacement
            branch = np.where(upper, "upper", "lower")
        else:
            means = np.where(upper, 0.545, 0.455)
            primary_y = rng.normal(means, 0.025)
            displacement = np.where(
                upper,
                np.exp(primary_y - 0.54) * continuous_time * 0.1,
                -np.exp(0.46 - primary_y) * continuous_time * 0.1,
            )
            secondary_y = primary_y + displacement
            branch = np.where(upper, "branch_A", "branch_B")
        primary[label] = np.column_stack([continuous_time, primary_y]).astype(np.float32)
        secondary[label] = np.column_stack([continuous_time, secondary_y]).astype(np.float32)
        branches[label] = np.asarray(branch)
    return primary, secondary, branches


def make_anndata(n_per_time=400, seed=42):
    """Generate paired observations and assemble canonical AnnData metadata."""
    import anndata as ad

    primary, secondary, branches = generate_branch_sync(n_per_time=n_per_time, seed=seed)
    x = np.concatenate([primary[label] for label in TIME_LABELS])
    y = np.concatenate([secondary[label] for label in TIME_LABELS])
    obs = pd.DataFrame({
        "time": np.repeat(TIME_POINTS, n_per_time),
        "celltype": np.concatenate([branches[label] for label in TIME_LABELS]),
        "barcode": [f"time_{stage}_cell_{cell:04d}"
                    for stage in range(5) for cell in range(n_per_time)],
    })
    obs.index = obs["barcode"].astype(str)
    cells = ad.AnnData(X=x.copy(), obs=obs)
    cells.obsm["X_latent"] = x
    cells.obsm["X_secondary"] = y
    cells.uns["branch_sync"] = {
        "generator_seed": int(seed),
        "cells_per_time": int(n_per_time),
        "time_labels": TIME_LABELS,
        "training_times": TIME_POINTS.tolist(),
        "physical_x_bins": TIME_EDGES.tolist(),
    }
    return cells


def evaluate_map(cells, model):
    """Return paired RMSE and R2 at each observed stage."""
    import torch

    rows = []
    for time in TIME_POINTS:
        mask = cells.obs["time"].to_numpy() == time
        x = torch.tensor(cells.obsm["X_latent"][mask], dtype=torch.float32)
        target = cells.obsm["X_secondary"][mask]
        with torch.no_grad():
            predicted = model(x, x.new_full((len(x),), float(time))).cpu().numpy()
        residual = np.square(predicted - target)
        denominator = np.square(target - target.mean(0, keepdims=True)).sum()
        rows.append({
            "Time": time,
            "Pairs": len(x),
            "RMSE": float(np.sqrt(residual.mean())),
            "R2": float(1.0 - residual.sum() / max(denominator, 1e-12)),
        })
    return pd.DataFrame(rows).set_index("Time")


def map_points(model, points, times):
    """Apply a frozen time-conditioned map to a trajectory array."""
    import torch

    q = torch.tensor(np.asarray(points), dtype=torch.float32)
    t = torch.tensor(np.broadcast_to(np.asarray(times)[:, None], q.shape[:2]).copy(),
                     dtype=q.dtype)
    with torch.no_grad():
        return model(q, t).cpu().numpy()


def plot_data_and_map(cells, model):
    """Show weak primary branching, amplified secondary branching and fitted T."""
    x = cells.obsm["X_latent"]
    y = cells.obsm["X_secondary"]
    times = cells.obs["time"].to_numpy(float)
    import torch
    with torch.no_grad():
        tx = model(torch.tensor(x, dtype=torch.float32),
                   torch.tensor(times, dtype=torch.float32)).cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), layout="constrained")
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.08, 0.88, len(TIME_POINTS)))
    for ax, values, title in zip(axes, (x, y, tx), ("Primary", "Secondary", r"$T$(Primary)")):
        for color, time in zip(colors, TIME_POINTS):
            mask = times == time
            ax.scatter(values[mask, 0], values[mask, 1], s=5, alpha=.42,
                       color=color, linewidths=0, rasterized=True, label=f"t={time:g}")
        ax.set(title=title, xlabel="Dimension 1", ylabel="Dimension 2",
               xlim=(-.03, 1.03), ylim=(.30, .72))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    axes[-1].legend(frameon=False, fontsize=8, ncol=1, loc="center left",
                    bbox_to_anchor=(1.02, .5))
    return fig


def plot_branch_comparison(cells, mapping, results, predictions, max_paths=160):
    """Compare individual balanced trajectories in primary and mapped spaces."""
    from matplotlib.collections import LineCollection

    observed_x = cells.obsm["X_latent"]
    observed_y = cells.obsm["X_secondary"]
    pick = np.linspace(0, predictions[next(iter(predictions))]["primary"].shape[1] - 1,
                       min(max_paths, 400), dtype=int)
    fig, axes = plt.subplots(len(results), 2, figsize=(9.2, 6.5),
                             sharex=True, sharey=True, layout="constrained")
    if len(results) == 1:
        axes = axes[None]
    for row, (name, result) in enumerate(results.items()):
        prediction = predictions[name]
        primary = prediction["primary"]
        mapped = map_points(mapping, primary, prediction["times"])
        path_colors = "#168BD2"
        for col, (values, observed, title) in enumerate(
                ((primary, observed_x, "Primary"), (mapped, observed_y, r"$T$(Primary)"))):
            ax = axes[row, col]
            ax.scatter(observed[:, 0], observed[:, 1], s=3, color="#b7bdc5",
                       alpha=.28, linewidths=0, rasterized=True)
            paths = values[:, pick].transpose(1, 0, 2)
            ax.add_collection(LineCollection(paths, colors=path_colors, linewidths=.65,
                                              alpha=.34, rasterized=True))
            ax.set(xlim=(-.03, 1.03), ylim=(.30, .72), xlabel="Dimension 1",
                   ylabel=name if col == 0 else "Dimension 2")
            if row == 0:
                ax.set_title(title)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
    return fig


def branch_diagnostics(cells, mapping, results, predictions):
    """Measure distributional branch separation and secondary endpoint matching."""
    import torch
    from geomloss import SamplesLoss

    target = torch.tensor(cells.obsm["X_secondary"][cells.obs["time"].to_numpy() == TIME_POINTS[-1]])
    synchronized = next(result for result in results.values()
                        if result.metadata["config"]["sync_loss"])
    blur = synchronized.metadata["blur_secondary"][-1]
    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=blur, backend="tensorized")
    rows = []
    for name, prediction in predictions.items():
        primary = prediction["primary"]
        mapped = map_points(mapping, primary, prediction["times"])
        gaps = []
        for values in (primary[-1, :, 1], mapped[-1, :, 1]):
            median = np.median(values)
            gaps.append(float(values[values >= median].mean() - values[values < median].mean()))
        rows.append({
            "Model": name,
            "Primary terminal separation": gaps[0],
            "Mapped terminal separation": gaps[1],
            "Secondary endpoint Sinkhorn": float(sinkhorn(torch.tensor(mapped[-1]), target)),
        })
    return pd.DataFrame(rows).set_index("Model")
