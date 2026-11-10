"""FiLM map architecture used by the archived pancreas checkpoint."""
from __future__ import annotations

import torch
from torch import nn


class FiLMMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=128, n_layers=3, t_hidden=32, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.n_modulated = max(n_layers - 1, 1)
        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_hidden), nn.GELU(),
            nn.Linear(t_hidden, 2 * hidden * self.n_modulated),
        )
        self.in_proj = nn.Linear(d_in, hidden)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden, hidden) for _ in range(self.n_modulated - 1)
        )
        self.out_proj = nn.Linear(hidden, d_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=q.dtype, device=q.device)
        if t.dim() == 0:
            t = t.expand(q.shape[:-1])
        film = self.t_encoder(t.to(q.dtype).unsqueeze(-1))
        film = film.view(*film.shape[:-1], self.n_modulated, 2, self.hidden)
        h = self.in_proj(q)
        h = self.dropout(self.activation(h * (1 + film[..., 0, 0, :]) + film[..., 0, 1, :]))
        for index, layer in enumerate(self.hidden_layers, start=1):
            h = layer(h)
            h = self.dropout(self.activation(
                h * (1 + film[..., index, 0, :]) + film[..., index, 1, :]
            ))
        return self.out_proj(h)


def load_film_checkpoint(path, *, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = FiLMMLP(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)
