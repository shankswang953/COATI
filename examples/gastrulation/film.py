"""Gastrulation FiLM map architecture and fitting recipe used by COATI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from geomloss import SamplesLoss
from torch import nn
from torch.nn import functional as F


TIMES = (0.0, 1.0, 2.0, 2.5)


class FiLMMLP(nn.Module):
    """Time-conditioned RNA-to-ATAC map used by the COATI gastrulation run."""

    def __init__(self, d_in, d_out, hidden=128, n_layers=3, t_hidden=32, dropout=0.0):
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
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _expand_t(self, t, q):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=q.dtype, device=q.device)
        if t.dim() == 0:
            t = t.to(q.dtype).expand(q.shape[:-1])
        else:
            t = t.to(q.dtype)
        return t.unsqueeze(-1)

    def forward(self, q, t):
        film = self.t_encoder(self._expand_t(t, q))
        film = film.view(*film.shape[:-1], self.n_modulated, 2, self.hidden)
        gammas = film[..., 0, :]
        betas = film[..., 1, :]
        h = self.in_proj(q)
        h = self.dropout(self.act(h * (1.0 + gammas[..., 0, :]) + betas[..., 0, :]))
        for index, layer in enumerate(self.hidden_layers):
            h = layer(h)
            h = self.dropout(self.act(
                h * (1.0 + gammas[..., index + 1, :]) + betas[..., index + 1, :]
            ))
        return self.out_proj(h)


def load_scale(path):
    scale = float(torch.load(path, map_location="cpu", weights_only=True)["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale in {path}")
    return scale


def load_snapshots(adata_path, primal_norm_path, secondary_norm_path, *, device="cpu"):
    """Load paired PCA/LSI snapshots and apply the two saved COATI scales."""
    import anndata as ad

    cells = ad.read_h5ad(adata_path)
    required = {"time", "stage", "celltype", "barcode"}
    if missing := required - set(cells.obs):
        raise KeyError(f"Missing AnnData observation columns: {sorted(missing)}")
    for key in ("X_pca", "X_atac"):
        if key not in cells.obsm:
            raise KeyError(f"Missing cells.obsm[{key!r}]")
    observed = tuple(float(value) for value in np.sort(cells.obs["time"].unique()))
    if observed != TIMES:
        raise ValueError(f"Expected gastrulation times {TIMES}, found {observed}")
    primary_scale = load_scale(primal_norm_path)
    secondary_scale = load_scale(secondary_norm_path)
    primary, secondary = [], []
    for time in TIMES:
        rows = cells.obs["time"].to_numpy() == time
        x = torch.tensor(np.asarray(cells.obsm["X_pca"])[rows],
                         dtype=torch.float32, device=device)
        y = torch.tensor(np.asarray(cells.obsm["X_atac"])[rows],
                         dtype=torch.float32, device=device)
        primary.append(x / primary_scale)
        secondary.append(y / secondary_scale)
    return primary, secondary


def _split_train_validation(primary, secondary, validation_fraction, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_primary, train_secondary, val_primary, val_secondary = [], [], [], []
    for x, y in zip(primary, secondary):
        n_validation = int(len(x) * validation_fraction)
        permutation = torch.randperm(len(x), generator=generator)
        train_rows = permutation[:-n_validation].to(x.device)
        val_rows = permutation[-n_validation:].to(x.device)
        train_primary.append(x[train_rows])
        train_secondary.append(y[train_rows])
        val_primary.append(x[val_rows])
        val_secondary.append(y[val_rows])
    return train_primary, train_secondary, val_primary, val_secondary


def _flatten_with_time(snapshots, device):
    values = torch.cat(snapshots)
    times = torch.cat([
        torch.full((len(snapshot),), TIMES[index], dtype=snapshot.dtype, device=device)
        for index, snapshot in enumerate(snapshots)
    ])
    return values, times


def _validation_sinkhorn(model, primary, secondary, losses, *, repeats, sample_size, seed):
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed + 9999)
    values = []
    with torch.no_grad():
        for index, (x, y) in enumerate(zip(primary, secondary)):
            n = min(sample_size, len(x))
            scores = []
            for _ in range(repeats):
                rows = torch.randperm(len(x), generator=generator)[:n].to(x.device)
                scores.append(losses[index](model(x[rows], TIMES[index]), y[rows]).item())
            values.append(float(np.mean(scores)))
    model.train()
    return values


def train_film_map(adata_path, output_path, primal_norm_path, secondary_norm_path, *,
                   device="cpu", epochs=300, seed=0, validation_fraction=.2,
                   learning_rate=1e-3, weight_decay=1e-4, batch_size=1024,
                   eval_every=5, eval_sample_size=1024, eval_repeats=10,
                   sinkhorn_blur_coeff=.05):
    """Run the original 300-epoch MSE fit and Sinkhorn checkpoint selection."""
    from coati._core.MapSpace import compute_per_time_median_distance

    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    primary, secondary = load_snapshots(
        adata_path, primal_norm_path, secondary_norm_path, device=device,
    )
    train_x, train_y, val_x, val_y = _split_train_validation(
        primary, secondary, validation_fraction, seed,
    )
    flat_x, flat_t = _flatten_with_time(train_x, device)
    flat_y, _ = _flatten_with_time(train_y, device)
    medians = compute_per_time_median_distance(secondary, n_sample=2000)
    losses = [
        SamplesLoss(loss="sinkhorn", p=2, blur=max(sinkhorn_blur_coeff * value, 1e-6)).to(device)
        for value in medians
    ]
    model = FiLMMLP(50, 14, hidden=128, n_layers=3, t_hidden=32, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_value, best_epoch, best_state = float("inf"), -1, None
    history = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(flat_x), generator=generator).to(device)
        total = 0.0
        for rows in permutation.split(batch_size):
            prediction = model(flat_x[rows], flat_t[rows])
            loss = F.mse_loss(prediction, flat_y[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        train_mse = total / len(flat_x)
        if epoch % eval_every == 0 or epoch == epochs:
            per_time = _validation_sinkhorn(
                model, val_x, val_y, losses, repeats=eval_repeats,
                sample_size=eval_sample_size, seed=seed,
            )
            value = float(sum(per_time))
            history.append({"epoch": epoch, "train_mse": train_mse,
                            "validation_sinkhorn": per_time, "validation_sum": value})
            print(f"epoch {epoch:3d}  train MSE {train_mse:.6g}  validation Sinkhorn sum {value:.6g}")
            if value < best_value:
                best_value, best_epoch = value, epoch
                best_state = {key: tensor.detach().cpu().clone()
                              for key, tensor in model.state_dict().items()}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "config": {"d_in": 50, "d_out": 14, "hidden": 128, "n_layers": 3,
                   "t_hidden": 32, "dropout": 0.0},
        "meta": {"epoch": best_epoch, "val_sum": best_value, "time_points": list(TIMES)},
    }
    torch.save(checkpoint, output_path)
    model.load_state_dict(best_state)
    return checkpoint, history


def load_film_checkpoint(path, *, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = FiLMMLP(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)
