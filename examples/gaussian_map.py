"""Original Gaussian-bump map architecture, fitting and held-out diagnostics.

All inputs are freshly prepared AnnData. No observations or pretrained weights
are loaded by this module. The fitting objective follows train_bump_T.py.
"""
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from generate_gaussian_data import gaussian_bump_z


class FourierBumpMLP(nn.Module):
    """
    T(q, t) -> (x, y, z). Time is accepted for SyncOT compatibility but ignored.

    Fourier features make the localized Gaussian peak easier to learn, while
    the small residual head keeps the first two coordinates close to identity.
    """

    def __init__(
        self,
        d_in=2,
        d_out=3,
        hidden=160,
        n_layers=4,
        num_frequencies=10,
        max_frequency=18.0,
        xy_residual_scale=0.05,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.hidden = hidden
        self.n_layers = n_layers
        self.num_frequencies = num_frequencies
        self.max_frequency = max_frequency
        self.xy_residual_scale = xy_residual_scale

        freqs = torch.logspace(
            0.0,
            np.log10(max_frequency),
            steps=num_frequencies,
            dtype=torch.float32,
        )
        self.register_buffer("freqs", freqs)

        feature_dim = d_in * (1 + 2 * num_frequencies)
        layers = [nn.Linear(feature_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, d_out)]
        self.net = nn.Sequential(*layers)

    def features(self, q_flat):
        parts = [q_flat]
        scaled = q_flat.unsqueeze(-1) * self.freqs.reshape(1, 1, -1) * np.pi
        parts.append(torch.sin(scaled).flatten(start_dim=1))
        parts.append(torch.cos(scaled).flatten(start_dim=1))
        return torch.cat(parts, dim=1)

    def forward(self, q, t=None):
        orig_shape = q.shape[:-1]
        q_flat = q.reshape(-1, self.d_in)
        raw = self.net(self.features(q_flat))
        xy = q_flat + self.xy_residual_scale * raw[:, :2]
        z = raw[:, 2:3]
        out = torch.cat([xy, z], dim=1)
        return out.reshape(*orig_shape, self.d_out)


def weighted_vector_loss(pred, target, sample_weight, z_weight):
    error = pred - target
    per_pair = error[:, :2].square().mean(1) + z_weight * error[:, 2].square()
    return (sample_weight * per_pair).sum() / (sample_weight.sum() + 1e-12)


def fit_bump_map(pairs, *, seed=0, steps=6000, batch_size=768, lr=1e-3,
                 weight_decay=1e-5, val_fraction=.15, z_weight=12.,
                 observed_weight=1., support_weight=1., eval_every=500,
                 hidden=160, n_layers=4, num_frequencies=10,
                 max_frequency=18., xy_residual_scale=.05, progress_path=None):
    """Fit the original Fourier map from freshly generated paired observations.

    A single seeded 85/15 split includes both observed pairs and bump-support
    pairs. Best-state selection uses validation loss only. Model parameters are
    frozen after fitting; differentiation with respect to its inputs is retained.
    """
    if pairs.n_obs < 3 or steps < 1 or not 0 < val_fraction < 1:
        raise ValueError("Provide paired observations, positive steps and a validation fraction in (0, 1).")
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.as_tensor(np.asarray(pairs.X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(pairs.obsm["X_secondary"]), dtype=torch.float32)
    kinds = pairs.obs["source"].astype(str).to_numpy()
    weights = torch.as_tensor(np.where(kinds == "Observed", observed_weight, support_weight), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(x), generator=generator)
    n_val = max(1, int(len(x) * val_fraction))
    train_idx, val_idx = permutation[n_val:], permutation[:n_val]
    split = np.full(len(x), "Training", dtype=object)
    split[val_idx.numpy()] = "Validation"
    pairs.obs["map_split"] = pd.Categorical(split)
    model = FourierBumpMLP(hidden=hidden, n_layers=n_layers,
        num_frequencies=num_frequencies, max_frequency=max_frequency,
        xy_residual_scale=xy_residual_scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, best_step = float("inf"), None, 0
    history = []
    for step in range(1, steps + 1):
        choice = train_idx[torch.randint(train_idx.numel(), (batch_size,))]
        loss = weighted_vector_loss(model(x[choice]), y[choice], weights[choice], z_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                prediction = model(x[val_idx])
                val_loss = float(weighted_vector_loss(prediction, y[val_idx], weights[val_idx], z_weight))
                z_rmse = float((prediction[:, 2] - y[val_idx, 2]).square().mean().sqrt())
            model.train()
            history.append(dict(step=step, training_loss=float(loss.detach()), validation_loss=val_loss,
                                validation_z_rmse=z_rmse))
            if val_loss < best_loss:
                best_loss, best_step = val_loss, step
                best_state = copy.deepcopy(model.state_dict())
            print(f"T step {step:4d}/{steps}: validation loss={val_loss:.6g}, z RMSE={z_rmse:.6g}", flush=True)
            if progress_path is not None:
                Path(progress_path).write_text(json.dumps(dict(stage="Train T", step=step, steps=steps,
                    best_step=best_step, best_validation_loss=best_loss)) + "\n")
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, pd.DataFrame(history), dict(best_step=best_step, best_validation_loss=best_loss,
        n_training=int(len(train_idx)), n_validation=int(len(val_idx)))


def evaluate_bump_map(model, pairs, *, grid_n=96, sigma=.08, amplitude=1.):
    """Evaluate validation pairs and a deterministic grid unused for fitting/selection."""
    axis = np.linspace(0., 1., grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    target = np.column_stack([grid, gaussian_bump_z(grid, sigma, amplitude)])
    with torch.no_grad():
        grid_prediction = model(torch.from_numpy(grid)).numpy()
        pair_prediction = model(torch.as_tensor(np.asarray(pairs.X), dtype=torch.float32)).numpy()
        center_prediction = float(model(torch.tensor([[.5, .5]]))[0, 2])
    rows = []
    validation = pairs.obs["map_split"].astype(str).to_numpy() == "Validation"
    kinds = pairs.obs["source"].astype(str).to_numpy()
    for name in ["Observed", "Support"]:
        mask = validation & (kinds == name)
        prediction = pair_prediction[mask]
        truth = np.asarray(pairs.obsm["X_secondary"])[mask]
        error = prediction - truth
        rows.append(dict(Evaluation=f"Held-out {name.lower()}", Pairs=int(mask.sum()),
            XY_RMSE=float(np.sqrt(np.mean(error[:, :2]**2))),
            Z_RMSE=float(np.sqrt(np.mean(error[:, 2]**2))), Z_MAE=float(np.mean(np.abs(error[:, 2])))))
    error = grid_prediction - target
    rows.append(dict(Evaluation="Independent grid", Pairs=len(grid),
        XY_RMSE=float(np.sqrt(np.mean(error[:, :2]**2))),
        Z_RMSE=float(np.sqrt(np.mean(error[:, 2]**2))), Z_MAE=float(np.mean(np.abs(error[:, 2])))))
    metrics = pd.DataFrame(rows).set_index("Evaluation")
    grid_evaluation = dict(xx=xx, yy=yy, grid=grid, target=target, prediction=grid_prediction,
        z_correlation=float(np.corrcoef(target[:, 2], grid_prediction[:, 2])[0, 1]),
        center_predicted_height=center_prediction, target_center_height=float(amplitude))
    return metrics, grid_evaluation


def plot_map_training(history):
    fig, ax = plt.subplots(figsize=(7.2, 3.3), layout="constrained")
    ax.semilogy(history.step, history.training_loss, color="#1688b1", label="Training minibatch")
    ax.semilogy(history.step, history.validation_loss, color="#f28e2b", label="Validation")
    ax.set(xlabel="Training step", ylabel="Weighted MSE", title="Train T from scratch")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return fig


def plot_map_performance(grid):
    shape = grid["xx"].shape
    truth = grid["target"][:, 2].reshape(shape)
    prediction = grid["prediction"][:, 2].reshape(shape)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), layout="constrained")
    low, high = min(0., float(prediction.min())), max(1., float(prediction.max()))
    for ax, values, title in zip(axes, [truth, prediction], ["Target bump", "Trained T"]):
        image = ax.imshow(values, origin="lower", extent=(0, 1, 0, 1),
                          vmin=low, vmax=high, cmap="viridis")
        ax.set(title=title, xlabel="Primary 1", ylabel="Primary 2", aspect="equal")
        fig.colorbar(image, ax=ax, shrink=.8)
    return fig
