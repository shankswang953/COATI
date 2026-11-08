"""Load the pretrained cross-modal maps used by the scMultiSim tutorials."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class _MLPWithTime(nn.Module):
    """MLP that appends time to the primary-space coordinates."""

    def __init__(self, d_in: int, d_out: int, hidden: int = 128,
                 n_layers: int = 3, dropout: float = 0.0, **_: object):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in + 1, hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _index in range(max(n_layers - 2, 0)):
            layers.extend((nn.Linear(hidden, hidden), nn.GELU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, time: torch.Tensor | float) -> torch.Tensor:
        time = torch.as_tensor(time, dtype=x.dtype, device=x.device)
        time = torch.broadcast_to(time, x.shape[:-1]).unsqueeze(-1)
        return self.net(torch.cat((x, time), dim=-1))


def load_trained_map(path: str | Path) -> nn.Module:
    """Load a frozen ``MLP + t`` or ``Ridge`` map checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    architecture = config["arch"]
    if architecture == "tmlp":
        model: nn.Module = _MLPWithTime(**config)
    elif architecture == "ridge":
        model = nn.Linear(config["d_in"], config["d_out"])
        checkpoint = {
            **checkpoint,
            "state_dict": {
                name.removeprefix("linear."): value
                for name, value in checkpoint["state_dict"].items()
            },
        }
    else:
        raise ValueError(f"Unsupported map architecture: {architecture!r}")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    return model
