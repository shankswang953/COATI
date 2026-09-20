"""Paired FiLM/MLP maps fitted only on the snapshots supplied to COATI."""
import copy
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import MapConfig


class MapQualityWarning(UserWarning):
    """Within-stage validation raises concerns about a learned cross-modal map."""


class PairedMap(nn.Module):
    """FiLM or ordinary MLP with training-only feature standardization.

    Inputs and outputs are in trajectory-training coordinates. FiLM accepts scalar
    times or times broadcastable to the leading dimensions of q. MLP ignores time.
    """
    def __init__(self, input_dim, output_dim, *, kind="film", hidden_dim=128,
                 n_layers=3, time_hidden_dim=32):
        super().__init__()
        if kind not in {"film", "mlp"}:
            raise ValueError("PairedMap kind must be 'film' or 'mlp'.")
        self.spec = dict(input_dim=input_dim, output_dim=output_dim, kind=kind,
                         hidden_dim=hidden_dim, n_layers=n_layers, time_hidden_dim=time_hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)] +
                                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)])
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        if kind == "film":
            self.time_encoder = nn.Sequential(nn.Linear(1, time_hidden_dim), nn.GELU(),
                                               nn.Linear(time_hidden_dim, 2 * hidden_dim * len(self.layers)))
            nn.init.zeros_(self.time_encoder[-1].weight)
            nn.init.zeros_(self.time_encoder[-1].bias)
        for name, value in (("input_mean", torch.zeros(input_dim)), ("input_std", torch.ones(input_dim)),
                            ("target_mean", torch.zeros(output_dim)), ("target_std", torch.ones(output_dim)),
                            ("time_start", torch.tensor(0.)), ("time_span", torch.tensor(1.))):
            self.register_buffer(name, value)

    def forward(self, q, t=None):
        h = (q - self.input_mean) / self.input_std
        if self.spec["kind"] == "film":
            if t is None:
                raise ValueError("A FiLM map requires a time input; use map_type='mlp' for a map without time.")
            t = torch.as_tensor(t, dtype=q.dtype, device=q.device)
            t = torch.broadcast_to(t, q.shape[:-1])
            modulation = self.time_encoder(((t - self.time_start) / self.time_span).unsqueeze(-1))
            modulation = modulation.reshape(*q.shape[:-1], len(self.layers), 2, self.spec["hidden_dim"])
        for index, layer in enumerate(self.layers):
            h = layer(h)
            if self.spec["kind"] == "film":
                h = h * (1 + modulation[..., index, 0, :]) + modulation[..., index, 1, :]
            h = self.activation(h)
        return self.output(h) * self.target_std + self.target_mean


class _TimeIndependentMap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, q, t=None):
        return self.model(q)


def freeze_external_map(model, device):
    """Copy a user map; adapt forward(q) without catching errors inside forward."""
    model = copy.deepcopy(model).to(device).eval()
    signature = inspect.signature(model.forward)
    try:
        signature.bind(None, None)
    except TypeError:
        try:
            signature.bind(None)
        except TypeError as exc:
            raise TypeError("T_model must accept forward(q) or forward(q, t).") from exc
        model = _TimeIndependentMap(model)
    model.requires_grad_(False)
    return model


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _distribution_scores(prediction, target, cfg, rng):
    """Compare equally sized validation samples using the same Sinkhorn settings."""
    from geomloss import SamplesLoss
    n = min(cfg.eval_samples, len(target) // 2)
    if n < 2:
        return dict(sinkhorn_divergence=None, observed_split_floor=None, sinkhorn_ratio=None,
                    sinkhorn_sample_size=n, regularization_length=None)
    cloud = target[rng.choice(len(target), min(len(target), 2 * cfg.eval_samples), replace=False)]
    length = max(0.05 * float(torch.pdist(cloud).median()), 1e-6)
    loss = SamplesLoss("sinkhorn", p=2, blur=length, debias=True, backend="tensorized")
    discrepancies, floors = [], []
    for _ in range(cfg.eval_repeats):
        indices = rng.permutation(len(target))[:2*n]
        a, b = indices[:n], indices[n:]
        discrepancies.append(float(loss(prediction[a], target[a])))
        floors.append(float(loss(target[a], target[b])))
    discrepancy = max(0., float(np.mean(discrepancies)))
    floor = max(0., float(np.mean(floors)))
    return dict(sinkhorn_divergence=discrepancy, observed_split_floor=floor,
                sinkhorn_ratio=discrepancy / floor if floor > 1e-12 else None,
                sinkhorn_sample_size=n, regularization_length=length)


def train_map(data, *, kind, config, directory, device, seed, primary_scale=1., secondary_scale=1., verbose=False):
    """Fit, validate, save and freeze a default map before trajectory fitting.

    No data discovery or full-atlas fallback is performed. All statistics and map
    fitting use training rows at supplied times. Validation rows select the checkpoint.
    """
    cfg = config if config is not None else MapConfig()
    if not isinstance(cfg, MapConfig):
        raise TypeError("map_config must be a MapConfig.")
    if data.secondary is None:
        raise ValueError("Map training requires paired secondary observations.")
    if any(len(x) < 4 for x in data.primary):
        raise ValueError("Map fitting needs at least four paired cells per supplied stage.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train, validation, splits = [], [], {}
    for k, (x, y, time) in enumerate(zip(data.primary, data.secondary, data.times)):
        indices = rng.permutation(len(x))
        n_val = max(2, min(len(x) - 2, int(round(len(x) * cfg.validation_fraction))))
        val_indices, train_indices = indices[:n_val], indices[n_val:]
        splits[f"train_{k}"] = train_indices
        splits[f"validation_{k}"] = val_indices
        for collection, rows in ((train, train_indices), (validation, val_indices)):
            collection.append((torch.tensor(x[rows] / primary_scale), torch.tensor(y[rows] / secondary_scale),
                               torch.full((len(rows),), float(time))))
    np.savez(directory / "splits.npz", **splits)
    _write_json(directory / "config.json", dict(kind=kind, seed=int(seed), **cfg.to_dict()))
    x, y, t = [torch.cat([stage[i] for stage in train]) for i in range(3)]
    vx, vy, vt = [torch.cat([stage[i] for stage in validation]) for i in range(3)]
    # The trajectory engine resets its own seed later; map batches have their own RNG.
    torch.manual_seed(seed)
    model = PairedMap(x.shape[1], y.shape[1], kind=kind, hidden_dim=cfg.hidden_dim,
                      n_layers=cfg.n_layers, time_hidden_dim=cfg.time_hidden_dim)
    model.input_mean.copy_(x.mean(0))
    model.input_std.copy_(x.std(0, unbiased=False).clamp_min(1e-6))
    model.target_mean.copy_(y.mean(0))
    model.target_std.copy_(y.std(0, unbiased=False).clamp_min(1e-6))
    model.time_start.fill_(float(data.times[0]))
    model.time_span.fill_(float(data.times[-1] - data.times[0]))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history, best_state, best_epoch, best_loss = [], None, 0, float("inf")
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.
        for rows in torch.randperm(len(x), generator=generator).split(cfg.batch_size):
            xb, yb, tb = x[rows].to(device), y[rows].to(device), t[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (((model(xb, tb) - yb) / model.target_std) ** 2).mean()
            if not torch.isfinite(loss):
                _write_json(directory / "report.json", dict(status="failed", reason="Non-finite map training loss", epoch=epoch))
                raise RuntimeError(f"Map training failed with a non-finite loss. See {directory / 'report.json'}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        model.eval()
        with torch.no_grad():
            val_sum = 0.
            for start in range(0, len(vx), cfg.batch_size):
                sl = slice(start, start + cfg.batch_size)
                error = (model(vx[sl].to(device), vt[sl].to(device)) - vy[sl].to(device)) / model.target_std
                val_sum += float(error.square().sum())
            val_loss = val_sum / vy.numel()
        if not np.isfinite(val_loss):
            _write_json(directory / "report.json", dict(status="failed", reason="Non-finite map validation loss", epoch=epoch))
            raise RuntimeError(f"Map validation failed. See {directory / 'report.json'}")
        history.append(dict(epoch=epoch, train_standardized_mse=total / len(x), validation_standardized_mse=val_loss))
        if verbose and (epoch == 1 or epoch % 25 == 0):
            print(f"[COATI map] Epoch {epoch}/{cfg.epochs}, standardized validation MSE {val_loss:.6g}.")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    torch.save(dict(format_version=1, spec=model.spec, state_dict=best_state), directory / "model.pt")
    _write_json(directory / "history.json", history)
    stages, issues = [], []
    with torch.no_grad():
        for k, ((tx, ty, tt), (valx, valy, valt)) in enumerate(zip(train, validation)):
            pred = torch.cat([model(valx[i:i+cfg.batch_size].to(device), valt[i:i+cfg.batch_size].to(device)).cpu()
                              for i in range(0, len(valx), cfg.batch_size)])
            mse = float((pred - valy).square().mean())
            baseline = float((ty.mean(0) - valy).square().mean())
            nmse = mse / baseline if baseline > 1e-12 else None
            row = dict(label=str(data.labels[k]), time=float(data.times[k]), train_cells=len(tx), validation_cells=len(valx),
                       paired_mse=mse, paired_rmse=float(np.sqrt(mse)), stage_mean_baseline_mse=baseline,
                       normalized_mse=nmse, **_distribution_scores(pred, valy, cfg, rng))
            stages.append(row)
            tag = f"{data.labels[k]} (t={data.times[k]:g})"
            if len(valx) < cfg.min_validation_cells:
                issues.append(f"{tag}: only {len(valx)} validation cells; quality estimates may be unreliable.")
            if nmse is None:
                issues.append(f"{tag}: near-zero baseline error; normalized paired error is undefined.")
            elif nmse > cfg.max_normalized_mse:
                issues.append(f"{tag}: normalized paired MSE {nmse:.3g} exceeds {cfg.max_normalized_mse:g}.")
            if row["sinkhorn_ratio"] is None:
                issues.append(f"{tag}: the observed split floor is unavailable or near zero; its ratio is undefined.")
            elif row["sinkhorn_ratio"] > cfg.max_sinkhorn_ratio:
                issues.append(f"{tag}: Sinkhorn/split-floor ratio {row['sinkhorn_ratio']:.3g} exceeds {cfg.max_sinkhorn_ratio:g}.")
    report = dict(status="warning" if issues else "passed", kind=kind, seed=int(seed),
                  training_times=data.times.tolist(), training_labels=[str(label) for label in data.labels],
                  primary_scale=primary_scale, secondary_scale=secondary_scale,
                  epochs_completed=len(history), best_epoch=best_epoch, best_validation_standardized_mse=best_loss,
                  validation_scope="Paired cells withheld within supplied stages; also used for checkpoint selection. Not an independent test of unseen stages or cell-type alignment.",
                  normalized_mse_definition="Paired validation MSE divided by validation MSE of the training-stage target mean; lower is better, 1 is the baseline.",
                  distribution_definition="Debiased Sinkhorn divergence with half squared Euclidean cost, using equal-sized validation subsamples. The split floor is an empirical reference, not a bound.",
                  thresholds=dict(max_normalized_mse=cfg.max_normalized_mse, max_sinkhorn_ratio=cfg.max_sinkhorn_ratio,
                                  min_validation_cells=cfg.min_validation_cells), stages=stages, warnings=issues)
    _write_json(directory / "report.json", report)
    lines = [f"COATI {kind} map validation: {report['status']}", f"Best epoch: {best_epoch}", report["validation_scope"], "",
             "Stage\tTime\tTrain\tValidation\tPaired RMSE\tNormalized MSE\tSinkhorn\tSplit floor"]
    for row in stages:
        lines.append("\t".join(str(row[key]) for key in ("label", "time", "train_cells", "validation_cells", "paired_rmse",
                                                         "normalized_mse", "sinkhorn_divergence", "observed_split_floor")))
    lines.extend(["", *issues])
    (directory / "report.txt").write_text("\n".join(lines) + "\n")
    if issues:
        message = "Cross-modal map validation raised concerns: " + " ".join(issues) + f" See {directory / 'report.txt'}."
        if cfg.quality_action == "error":
            raise RuntimeError(message + " Trajectory fitting was not started.")
        warnings.warn(message, MapQualityWarning, stacklevel=2)
    return model, report


def load_map(path, *, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported map checkpoint version.")
    model = PairedMap(**checkpoint["spec"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)
