import torch
import numpy as np
from sklearn.cluster import KMeans

import torch
import numpy as np
from sklearn.cluster import KMeans


class ConformalMetric:
    """
    Metric Flow Matching metric (diagonal / anisotropic version)

        g(x) = diag(M_1(x), ..., M_D(x))
        M_j(x) = (h_j(x) + eps)^(-alpha)
    """

    def __init__(
        self,
        X: torch.Tensor,
        n_centers: int = 50,
        kappa: float = 2.0,
        alpha: float = 1.0,
        eps: float = 1e-4,
        device: torch.device = torch.device('cpu'),
    ):
        self.alpha = alpha
        self.eps = eps

        # -------- Step 1: KMeans on CPU (sklearn requires numpy) --------
        X_np = X.detach().cpu().numpy()
        N, D = X_np.shape

        kmeans = KMeans(n_clusters=n_centers, n_init=10)
        kmeans.fit(X_np)
        centers = kmeans.cluster_centers_        # (K, D)
        labels = kmeans.labels_

        # -------- Step 2: estimate RBF bandwidths --------
        sigmas = []
        for k in range(n_centers):
            pts = X_np[labels == k]
            if len(pts) == 0:
                sigmas.append(1.0)
            else:
                var = np.mean(np.sum((pts - centers[k]) ** 2, axis=1))
                sigmas.append(np.sqrt(var + 1e-8))
        sigmas = np.array(sigmas).reshape(-1, 1)  # (K, 1)

        # store as torch tensors (buffers)
        self.C = torch.tensor(centers, dtype=torch.float32)                 # (K, D)
        self.lam = torch.tensor(0.5 / (kappa * sigmas) ** 2, dtype=torch.float32)  # (K, 1)

        # -------- Step 3: solve for ω (K × D) via least squares --------
        X_t = torch.tensor(X_np, dtype=torch.float32)   # (N, D)
        dist2 = torch.cdist(X_t, self.C) ** 2            # (N, K)

        Phi = torch.exp(
            -self.lam[None, :, :] * dist2[:, :, None]
        ).squeeze(-1)                                    # (N, K)

        ones = torch.ones(N, D)                           # (N, D)

        # ω ∈ ℝ^{K × D}
        self.w, *_ = torch.linalg.lstsq(Phi, ones)
        self.to(device)
        
    def to(self, device):
        self.C = self.C.to(device)
        self.lam = self.lam.to(device)
        if hasattr(self, "w"):
            self.w = self.w.to(device)
        return self

    # ------------------------------------------------------------------
    # Support function h(x): (B, D)
    # ------------------------------------------------------------------
    def h(self, x: torch.Tensor) -> torch.Tensor:
        C = self.C
        lam = self.lam
        w = self.w

        dist2 = torch.cdist(x, C) ** 2                   # (B, K)
        phi = torch.exp(
            -lam[None, :, :] * dist2[:, :, None]
        ).squeeze(-1)                                    # (B, K)

        s = phi @ w                      # (B, D)
        hx = torch.nn.functional.softplus(s) - np.log(2.0)   # softplus(0)=log(2) 

        return hx    # (B, D)

    # ------------------------------------------------------------------
    # Metric M(x): (B, D)
    # ------------------------------------------------------------------
    def metric(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.h(x)
        return 1.0 / (hx + self.eps) ** self.alpha

    # ------------------------------------------------------------------
    # Riemannian speed ||v||_{g(x)}
    # ------------------------------------------------------------------
    def speed(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        Mx = self.metric(x)                               # (B, D)
        return torch.sqrt((Mx * v ** 2).sum(dim=-1))      # (B,)