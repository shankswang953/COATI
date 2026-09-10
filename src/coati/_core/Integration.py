import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
from .DataLoad import load_source_data
import torch
import torch.optim as optim

import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import numpy as np

def plot_losses(log_file="loss_log.npz", out_file="loss_plot.png"):
    data = np.load(log_file)
    epochs = range(1, len(data["total"]) + 1)

    # 要画的 loss 名字和标题
    loss_keys = ["rec", "cross", "kl", "align", "smooth", "total"]
    loss_titles = [
        "Reconstruction Loss",
        "Cross-modal Loss",
        "KL Divergence",
        "Latent Alignment",
        "Smoothness Loss",
        "Total Loss"
    ]

    n_losses = len(loss_keys)
    ncols = 2
    nrows = int(np.ceil(n_losses / ncols))

    plt.figure(figsize=(12, 3 * nrows))

    for i, (key, title) in enumerate(zip(loss_keys, loss_titles), 1):
        plt.subplot(nrows, ncols, i)
        plt.plot(epochs, data[key], label=title, color="C0" if key!="total" else "black")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved subplot loss curves to {out_file}")

# -----------------------------
# Dataset class
# -----------------------------
class PairedDataset(Dataset):
    """
    Paired dataset of RNA (primary modality) and ATAC (secondary modality).
    """
    def __init__(self, x_rna, x_atac):
        self.x_rna = x_rna
        self.x_atac = x_atac

    def __len__(self):
        return self.x_rna.shape[0]

    def __getitem__(self, idx):
        return self.x_rna[idx], self.x_atac[idx]


# -----------------------------
# Network building blocks
# -----------------------------
def mlp(in_dim, hidden, out_dim, dropout=0.1):
    """Simple 3-layer MLP block."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, out_dim)
    )


class GaussianEncoder(nn.Module):
    """Encoder that outputs mean and log variance of latent z."""
    def __init__(self, in_dim, hidden, z_dim):
        super().__init__()
        self.backbone = mlp(in_dim, hidden, 2 * z_dim)
        self.z_dim = z_dim

    def forward(self, x):
        h = self.backbone(x)
        mu, logvar = h[..., :self.z_dim], h[..., self.z_dim:]
        return mu, logvar


class Decoder(nn.Module):
    """Decoder that reconstructs modality from latent z."""
    def __init__(self, z_dim, hidden, out_dim):
        super().__init__()
        self.net = mlp(z_dim, hidden, out_dim)

    def forward(self, z):
        return self.net(z)


def reparameterize(mu, logvar, training=True):
    """Reparameterization trick."""
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu


# -----------------------------
# Cross-modal VAE model
# -----------------------------
class CrossModalVAE(nn.Module):
    def __init__(self, n_rna, n_atac, z_dim=32, hidden=256):
        super().__init__()
        self.enc_r = GaussianEncoder(n_rna, hidden, z_dim)
        self.enc_a = GaussianEncoder(n_atac, hidden, z_dim)
        self.dec_r = Decoder(z_dim, hidden, n_rna)
        self.dec_a = Decoder(z_dim, hidden, n_atac)

    def encode_r(self, x_r, training=True):
        mu, logvar = self.enc_r(x_r)
        z = reparameterize(mu, logvar, training)
        return z, mu, logvar

    def encode_a(self, x_a, training=True):
        mu, logvar = self.enc_a(x_a)
        z = reparameterize(mu, logvar, training)
        return z, mu, logvar

    # Differentiable mapping RNA -> ATAC
    def map_rna_to_atac(self, x_r, training=False):
        z, _, _ = self.encode_r(x_r, training=training)
        return self.dec_a(z)

    def forward(self, x_r, x_a):
        z_r, mu_r, lv_r = self.encode_r(x_r, training=self.training)
        z_a, mu_a, lv_a = self.encode_a(x_a, training=self.training)

        xr_rec = self.dec_r(z_r)
        xa_rec = self.dec_a(z_a)

        xa_from_r = self.dec_a(z_r)
        xr_from_a = self.dec_r(z_a)

        return xr_rec, xa_rec, xa_from_r, xr_from_a, (mu_r, lv_r, mu_a, lv_a)


# -----------------------------
# Loss functions
# -----------------------------
def kl_loss(mu, logvar):
    """KL divergence between q(z|x) and N(0,I)."""
    return 0.5 * torch.mean(torch.sum(
        torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1))


def compute_main_loss(model, x_r, x_a,
                      w_rec=1.0, w_cross=1.0, w_kl=1e-3, w_align=1.0):
    """Core loss without smoothness."""
    xr_rec, xa_rec, xa_from_r, xr_from_a, (mu_r, lv_r, mu_a, lv_a) = model(x_r, x_a)

    l_rec = F.l1_loss(xr_rec, x_r) + F.l1_loss(xa_rec, x_a)
    l_cross = F.l1_loss(xa_from_r, x_a) + F.l1_loss(xr_from_a, x_r)
    l_kl = kl_loss(mu_r, lv_r) + kl_loss(mu_a, lv_a)
    l_align = F.mse_loss(mu_r, mu_a)
    
    print("l_rec: ", l_rec)
    print("l_cross: ", l_cross)
    print("l_kl: ", l_kl)
    print("l_align: ", l_align)

    return w_rec * l_rec + w_cross * l_cross + w_kl * l_kl + w_align * l_align


# -----------------------------
# kNN neighborhood construction
# -----------------------------
def build_knn_pairs(x_rna, k=5):
    """
    Build global kNN neighbor pairs from RNA space.
    Returns tensor of shape [num_pairs, 2].
    """
    X = x_rna.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    _, indices = nbrs.kneighbors(X)

    pairs = []
    for i in range(indices.shape[0]):
        for j in indices[i, 1:]:  # skip self (j=0)
            pairs.append((i, j))
    return torch.tensor(pairs, dtype=torch.long)


def smooth_loss(model, x_rna_all, neighbor_pairs, device, num_sample=500):
    """
    Neighborhood smoothness loss:
    Ensure that nearby RNA points map to nearby ATAC points.
    """
    idx = torch.randint(0, neighbor_pairs.shape[0], (min(num_sample, neighbor_pairs.shape[0]),))
    pairs = neighbor_pairs[idx]

    xi = x_rna_all[pairs[:, 0]].to(device)
    xj = x_rna_all[pairs[:, 1]].to(device)

    fi = model.map_rna_to_atac(xi, training=True)
    fj = model.map_rna_to_atac(xj, training=True)

    return F.mse_loss(fi, fj)


# -----------------------------
# Training loop
# -----------------------------
def train(model, loader, x_rna_all, neighbor_pairs,
          epochs=50, lr=1e-3, device='cuda',
          w_smooth=1e-2, log_file="loss_log.npz"):

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 用来存 loss
    history = {"rec": [], "cross": [], "kl": [], "align": [], "smooth": [], "total": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_rec, total_cross, total_kl, total_align, total_smooth = 0, 0, 0, 0, 0

        for x_r, x_a in loader:
            x_r, x_a = x_r.to(device), x_a.to(device)

            # forward & main loss
            xr_rec, xa_rec, xa_from_r, xr_from_a, (mu_r, lv_r, mu_a, lv_a) = model(x_r, x_a)

            l_rec = F.l1_loss(xr_rec, x_r) + F.l1_loss(xa_rec, x_a)
            l_cross = F.l1_loss(xa_from_r, x_a) + F.l1_loss(xr_from_a, x_r)
            l_kl = kl_loss(mu_r, lv_r) + kl_loss(mu_a, lv_a)
            l_align = F.mse_loss(mu_r, mu_a)
            l_smooth = smooth_loss(model, x_rna_all, neighbor_pairs, device)

            loss = l_rec + l_cross + 1e-3*l_kl + l_align + w_smooth*l_smooth

            opt.zero_grad()
            loss.backward()
            opt.step()

            # accumulate
            total_loss += loss.item()
            total_rec += l_rec.item()
            total_cross += l_cross.item()
            total_kl += l_kl.item()
            total_align += l_align.item()
            total_smooth += l_smooth.item()

        # 计算平均
        n_batches = len(loader)
        history["rec"].append(total_rec / n_batches)
        history["cross"].append(total_cross / n_batches)
        history["kl"].append(total_kl / n_batches)
        history["align"].append(total_align / n_batches)
        history["smooth"].append(total_smooth / n_batches)
        history["total"].append(total_loss / n_batches)

        print(f"Epoch {epoch}: "
              f"rec={history['rec'][-1]:.4f}, "
              f"cross={history['cross'][-1]:.4f}, "
              f"kl={history['kl'][-1]:.4f}, "
              f"align={history['align'][-1]:.4f}, "
              f"smooth={history['smooth'][-1]:.4f}, "
              f"total={history['total'][-1]:.4f}")

    np.savez(log_file, **history)
    return model, history

class MapGaussian(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, hidden=256):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

    def map_rna_to_atac(self, x, training=False):
        with torch.no_grad():
            return self.forward(x)



class AlignMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

    def map_rna_to_atac(self, x):
        return self.forward(x)
