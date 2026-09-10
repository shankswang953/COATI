from heapq import nlargest
from itertools import tee
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchdiffeq import odeint


# ============================================================
# Per-time median pairwise distance — shared low-level utility.
#
# Used by:
#   1. estimate_sigma_within_time   (Gaussian-kernel sigma)
#   2. adaptive Sinkhorn blur in run_training (Training.py)
#
# Memory hygiene: subsample BEFORE cdist, free intermediate
# tensors after each timepoint via explicit del.
# ============================================================
def compute_per_time_median_distance(data_list, n_sample=500):
    """
    Per-time pairwise-distance quantile (used as the data-adaptive scale
    for Sinkhorn blur and kernel bandwidth).

    For each timepoint, subsample ``n_sample`` cells, compute all
    pairwise distances among them (excluding self-distances), and take
    a quantile (the QUANTILE constant below, default 0.5 = median).

    Two natural choices for QUANTILE:

      - 0.5  -> median pairwise distance. The classical MMD median
               heuristic, also widely used as the data-adaptive Sinkhorn
               blur scale (POT, geomloss-derived recipes).
      - 0.95 -> a robust upper bound on typical pair distances. Some
               Sinkhorn pipelines use this for numerical stability so
               that exp(-d^2 / (2*blur^2)) does not underflow for the
               vast majority of pairs.

    Edit QUANTILE inline to switch (kept as an internal constant rather
    than a public arg to avoid changing the function signature -- many
    downstream call sites already invoke this without an explicit
    quantile argument).

    NOTE on scale change (2026): this function previously used the median
    k-NN (k = 20) distance, the LOCAL manifold scale. Pairwise quantile
    distance is typically LARGER than the k-NN scale, especially in
    high dimensions under concentration of measure. The downstream
    ``blur_coeff`` may therefore need to be retuned (typically reduce
    by ~3-5x) when switching from the old k-NN scale to the new
    pairwise scale to keep the absolute blur in the same regime.

    Args:
        data_list: list of (N_t, d) tensors, one per timepoint.
        n_sample:  subsample size per timepoint. Memory cost of cdist is
                   O(n_sample^2); 2000 fits easily, 5000 still OK.

    Returns:
        list[float]: per-timepoint quantile pairwise distance.
    """
    # ----------------------------------------------------------------
    # Internal knob -- flip between median heuristic (0.5) and
    # robust-upper-bound (0.95) without changing the function signature.
    # ----------------------------------------------------------------
    QUANTILE = 0.5

    if not (0.0 <= QUANTILE <= 1.0):
        raise ValueError(
            f"QUANTILE must be in [0, 1]; got {QUANTILE}")

    out = []
    for X_t in data_list:
        n_avail = X_t.shape[0]
        n_take = min(n_sample, n_avail)
        if n_take < 2:
            raise ValueError(
                f"Need at least 2 cells per timepoint to compute "
                f"pairwise distance; got {n_take}.")
        idx = torch.randperm(n_avail, device=X_t.device)[:n_take]
        sample = X_t[idx]
        d = torch.cdist(sample, sample)  # (n_take, n_take)

        # Strict upper triangle: all unique pairs (i < j), excluding
        # self-distances (diagonal) and duplicate pairs (i, j) vs (j, i).
        # This keeps the quantile from being biased toward 0 by the
        # n_take zero-distance diagonal entries.
        mask = torch.triu(
            torch.ones_like(d, dtype=torch.bool), diagonal=1)
        pair_d = d[mask]

        out.append(torch.quantile(pair_d, QUANTILE).item())

        # Free GPU memory immediately -- d is n_sample x n_sample.
        del d, pair_d, mask, sample, idx
    return out


# ============================================================
# Bandwidth estimator for the Gaussian-kernel projection.
# Reuses compute_per_time_median_distance to avoid duplicate work.
# ============================================================
def estimate_sigma_within_time(primary_data, n_sample=500):
    """
    Estimate the Gaussian-kernel bandwidth from within-time pairwise
    distances of a small subsample.

    Why within-time (not global): the kernel projection is meant to find
    cell-type-similar neighbors. Within-time distances capture that
    scale; between-time distances are typically larger because cells
    move with development, and a global median would inflate sigma.

    Args:
        primary_data: list of (N_t, d) tensors, one per time point.
        n_sample:     subsample size per time (default 500).

    Returns:
        sigma (float)
    """
    medians = compute_per_time_median_distance(primary_data, n_sample)
    return sum(medians) / len(medians) / 2


# ============================================================
# Per-cell adaptive bandwidth (used by map_type='accurate_kernel').
#
# For each cell j in the primary basis, sigma_j is set to the distance
# from cell j to its k-th nearest neighbor within the same timepoint.
# This makes the Gaussian kernel adapt to local density: dense regions
# get small sigma (sharp), sparse regions get large sigma (smooth).
#
# Theoretical motivation: see TraInf.tex Section 7.6 — the per-basis
# sigma generalization of Nadaraya-Watson kernel projection. Smaller
# sigma where cells are dense produces a tighter cross-modal estimate;
# larger sigma where cells are sparse compensates for less support.
#
# Memory hygiene: chunked cdist + topk per timepoint, results moved to
# CPU after computation to avoid GPU OOM with large N_t.
# ============================================================
def compute_per_cell_sigma_per_time(data_list, k=20, chunk_size=2000,
                                    device='cpu'):
    """
    For each timepoint t and each cell j in data_list[t], compute
        sigma_j = distance from cell j to its k-th nearest neighbor
                  (within the same timepoint, self excluded).

    Args:
        data_list:  list of (N_t, d) tensors, one per timepoint.
        k:          neighborhood size (default 20). If a timepoint has
                    fewer than k+1 cells, k is capped to N_t - 1.
        chunk_size: query batch size for chunked cdist; trades memory
                    against speed (default 2000).
        device:     device on which to run cdist; results are returned
                    on CPU regardless.

    Returns:
        list[Tensor]: per-timepoint cpu tensors of shape (N_t,), each
                      holding the per-cell k-th NN distance.
    """
    sigmas = []
    for t_idx, X in enumerate(data_list):
        N = X.shape[0]
        Xd = X.to(device)
        k_eff = min(k, N - 1) if N > 1 else 1
        if k_eff < k:
            print(f"[compute_per_cell_sigma] timepoint {t_idx}: only {N} "
                  f"cells, k capped from {k} to {k_eff}")
        sigma_t = torch.empty(N)  # cpu by default
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = Xd[start:end]
            d = torch.cdist(chunk, Xd)  # (chunk, N)
            # Mask self-distance: row i corresponds to global row start+i.
            local_idx = torch.arange(end - start, device=device)
            global_idx = torch.arange(start, end, device=device)
            d[local_idx, global_idx] = float('inf')
            knn_d, _ = torch.topk(d, k_eff, largest=False, dim=1)
            # k-th NN distance = last column of topk(k_eff, smallest first).
            sigma_t[start:end] = knn_d[:, -1].cpu()
            del d, knn_d
        sigmas.append(sigma_t)
        del Xd
    return sigmas


def clip_sigma_globally(sigmas_per_time, low_q=0.05, high_q=0.95):
    """
    Compute global percentile bounds across ALL timepoints and ALL cells,
    then clip each timepoint's sigma tensor to [p_low, p_high].

    Why global (not per-time): the temporal variation of sigma is itself
    informative (early timepoints may be denser than late ones). Per-time
    clipping would erase this signal. Global percentile bounds suppress
    only the tails (numerical outliers and isolated cells) while keeping
    inter-timepoint differences intact.

    Args:
        sigmas_per_time: list of (N_t,) cpu tensors from
                         compute_per_cell_sigma_per_time.
        low_q:           lower percentile in [0, 1] (default 0.05).
        high_q:          upper percentile in [0, 1] (default 0.95).

    Returns:
        (clipped_list, lo, hi):
            clipped_list: list of (N_t,) tensors with values clamped.
            lo:           low_q quantile of the global pool (float).
            hi:           high_q quantile of the global pool (float).
    """
    pool = torch.cat([s.flatten() for s in sigmas_per_time])
    lo = torch.quantile(pool, low_q).item()
    hi = torch.quantile(pool, high_q).item()
    clipped = [s.clamp(min=lo, max=hi) for s in sigmas_per_time]
    return clipped, lo, hi


# ============================================================
# Sinkhorn noise floor estimator.
#
# For each timepoint, the floor is the Sinkhorn divergence between two
# i.i.d. random subsamples drawn from the same empirical distribution.
# Trainable Sinkhorn(predicted, data) has this value as a lower bound:
# even a perfectly-matched model cannot drive Sinkhorn below the floor
# because the predicted batch and the data batch are themselves finite
# samples and remain distinguishable at the sample-noise level.
#
# Used as the dual-update target tau in the Lagrangian formulation of
# adaptive Sinkhorn weighting (see Training.py / Epoch.py).
# ============================================================
def compute_noise_floor_per_time(data, blurs, n_subsample, device,
                                 seed=0, n_repeats=50):
    """
    Estimate Sinkhorn noise floor per timepoint, returning both mean and std.

    For each timepoint t, draw two i.i.d. random subsamples (each of size
    n_subsample) from data[t] and compute Sinkhorn(batch1, batch2). Repeat
    n_repeats times. The mean over repeats is the expected lower bound for
    training Sinkhorn; the std measures the per-batch variance, which can
    be used to relax the dual target (e.g., target = mean + k * std) so
    training is not penalized for fluctuations within the natural noise
    band.

    Args:
        data:        list of (N_t, d) tensors, one per timepoint.
        blurs:       list of float per timepoint; the Sinkhorn blur to use
                     at each timepoint (typically args.blur_per_time_*).
        n_subsample: subsample size per batch. Should match the training
                     batch size (args.num_samples) so the floor reflects
                     what training will see.
        device:      torch device (used to instantiate SamplesLoss).
        seed:        RNG seed for reproducibility.
        n_repeats:   number of (batch1, batch2) repeats. Default 50 gives
                     a stable mean and a reasonable std estimate.

    Returns:
        list of dict, one per timepoint, each with keys:
            'mean': float, average Sinkhorn over n_repeats
            'std':  float, sample std over n_repeats
    """
    from geomloss import SamplesLoss

    torch.manual_seed(seed)
    results = []
    for t_idx, X_t in enumerate(data):
        sinkhorn = SamplesLoss(loss='sinkhorn', p=2,
                               blur=blurs[t_idx]).to(device)
        N = X_t.shape[0]
        # Cap n_subsample so two non-overlapping batches fit (best-effort;
        # we allow overlap if N is small). Using two independent randperms
        # may overlap when n_subsample > N/2, but that is acceptable.
        n_take = min(n_subsample, N)
        if n_take < 16:
            print(f"[Warn] timepoint {t_idx}: only {N} cells, n_take={n_take} "
                  f"(less than 16, floor estimate may be unstable)")
        repeats = []
        for _ in range(n_repeats):
            idx1 = torch.randperm(N, device=X_t.device)[:n_take]
            idx2 = torch.randperm(N, device=X_t.device)[:n_take]
            repeats.append(sinkhorn(X_t[idx1], X_t[idx2]).item())
        repeats_t = torch.tensor(repeats)
        results.append({
            'mean': repeats_t.mean().item(),
            'std':  repeats_t.std().item(),
        })
    return results


# ============================================================
# kernel_project_paired: differentiable, non-parametric mapping
# from primary to secondary space using paired training data
# (Gaussian kernel regression / Nadaraya-Watson estimator).
# ============================================================
def kernel_project_paired(traj_points, X_basis, Y_basis, sigma):
    """
    For each query point in `traj_points` (in primary space), compute
    Gaussian similarity weights against `X_basis` and apply those weights
    to the paired `Y_basis` to produce a secondary-space projection.

    Two bandwidth modes are supported through a single argument:

      (a) Isotropic (scalar sigma):
          weight[i, j] = softmax_j(-||q_i - X_basis[j]||^2 / (2 * sigma^2))

      (b) Per-basis adaptive (1-D tensor sigma of shape (N,)):
          weight[i, j] = softmax_j(-||q_i - X_basis[j]||^2 / (2 * sigma_j^2))
          Each basis cell j has its own bandwidth sigma_j (typically the
          k-th NN distance for that cell). See compute_per_cell_sigma_per_time
          and TraInf.tex Section 7.6 for the derivation.

    Fully differentiable: gradients flow from y_pred through the softmax
    weights back to traj_points (and so to upstream model parameters).

    Args:
        traj_points: (..., d_x) any-shape query batch in primary space.
        X_basis:     (N, d_x)   paired primary basis cells (detached).
        Y_basis:     (N, d_y)   paired secondary basis cells (same row
                                order as X_basis).
        sigma:       scalar (Python float / 0-D tensor) for isotropic mode,
                     OR 1-D tensor of shape (N,) for per-basis adaptive
                     mode (must match X_basis row count).

    Returns:
        Tensor of shape (..., d_y) — projection in secondary space, with
        the same leading batch dimensions as ``traj_points``.

    Notes:
        - X_basis / Y_basis must have the same first dimension N and be
          row-aligned (X_basis[j] and Y_basis[j] are the same cell).
        - X_basis / Y_basis are typically detached so they don't receive
          gradients (they are reference data, not parameters).
        - Per-basis sigma is also detached in practice (it is precomputed
          from data, not optimized).
        - Memory cost is O(B * N) where B = total query points.
    """
    orig_shape = traj_points.shape[:-1]
    d_x = traj_points.shape[-1]

    # Flatten any leading batch dims into a single B axis.
    q = traj_points.reshape(-1, d_x)                          # (B, d_x)

    # Pairwise squared distance in primary space.
    d_sq = torch.cdist(q, X_basis).pow(2)                     # (B, N)

    # Dispatch on sigma type:
    #   1-D tensor with size matching N -> per-basis bandwidth, broadcast
    #     (B, N) / (N,) row-wise (each column j divided by 2 * sigma_j^2).
    #   Otherwise -> isotropic (scalar) bandwidth.
    if torch.is_tensor(sigma) and sigma.dim() >= 1:
        if sigma.shape[0] != X_basis.shape[0]:
            raise ValueError(
                f"per-basis sigma length ({sigma.shape[0]}) must match "
                f"X_basis row count ({X_basis.shape[0]})"
            )
        sigma2 = sigma.to(d_sq.device).pow(2)                 # (N,)
        log_w = -d_sq / (2.0 * sigma2)                        # (B, N) by broadcast
    else:
        sigma_scalar = float(sigma) if torch.is_tensor(sigma) else sigma
        log_w = -d_sq / (2.0 * sigma_scalar * sigma_scalar)   # (B, N)

    # Softmax over basis: numerically stable (uses log-space max-subtract internally).
    weights = torch.softmax(log_w, dim=1)                     # (B, N)

    # Weighted average of paired secondary cells.
    y_pred = weights @ Y_basis                                # (B, d_y)

    # Restore the original leading batch shape.
    return y_pred.reshape(*orig_shape, Y_basis.shape[-1])


# ============================================================
# AlignRidge: a frozen, differentiable PyTorch wrapper.
# Single global linear map, forward signature matches AlignMLP.
# ============================================================
class AlignRidge(nn.Module):
    """
    y = x @ W + b

    Buffers (not Parameters) hold the fitted weights, so:
      - they ARE saved/loaded via state_dict()
      - they are NOT updated by an optimizer
      - gradients still flow through (matmul is differentiable),
        which is what you want when this module sits inside a
        larger trainable pipeline.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.register_buffer('W', torch.zeros(in_dim, out_dim))
        self.register_buffer('b', torch.zeros(out_dim))
        self.register_buffer('best_alpha', torch.tensor(0.0))

    def forward(self, x):
        return x @ self.W + self.b

    @torch.no_grad()
    def set_weights(self, W_np, b_np, alpha):
        self.W.copy_(torch.tensor(W_np, dtype=self.W.dtype))
        self.b.copy_(torch.tensor(b_np, dtype=self.b.dtype))
        self.best_alpha.fill_(float(alpha))



def map_whole_trajectory2another_manifold(
    traj_points,
    support_points_main,
    support_points_aux,
    k=20,
    alpha_main=100.0,
    sigma=0.05,
    type = "accrossSpace"
):
    if type == "time_terminalMatching":
        return _map_whole_trajectory2another_manifold_time_terminalMatching(traj_points, support_points_main, support_points_aux, k, alpha_main, sigma)
    elif type == "attention":
        return _map_whole_trajectory2another_manifold_attention(traj_points, support_points_main, support_points_aux, k, alpha_main, sigma)
    elif type == "accrossSpace":
        return _map_whole_trajectory2another_manifold_accrossSpace(traj_points, support_points_main, support_points_aux, k, sigma)
    elif type == "Simple":
        return _map_whole_trajectory2another_manifold_Simple(traj_points, support_points_main, support_points_aux, k, sigma)
    else:
        raise ValueError(f"Invalid type: {type}")




def map_to_nearest_manifold(sample_points, manifold_points, k=10, sigma=0.05, max_distance=0.05):
    """
    Map trajectory points to manifold points
    
    Args:
        sample_points: tensor of shape (n_samples, dim) - points from trajectory
        manifold_points: tensor of shape (n_manifold, dim) - points on manifold
        k: number of nearest neighbors (default: 5)
        sigma: kernel bandwidth (default: 0.1)
        max_distance: maximum allowed distance for mapping (default: 0.1)
    
    Returns:
        mapped_points: tensor of shape (n_samples, dim)
        indices: tensor of shape (n_samples, k)
        kernel_weights: tensor of shape (n_samples, k)
    """
    # Calculate pairwise distances
    dist_matrix = torch.cdist(sample_points, manifold_points)
    
    # Get k-nearest neighbors
    distances, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
    
    # Apply distance threshold
    if max_distance is not None:
        # Create a mask for distances within threshold
        valid_mask = distances <= max_distance
        # Set large weights for invalid distances
        distances = torch.where(valid_mask, distances, torch.ones_like(distances) * 1e6)
    
    # Compute Gaussian kernel weights
    kernel_weights = torch.exp(-distances**2 / (2 * sigma**2))
    
    # Normalize weights
    kernel_weights = kernel_weights / (kernel_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Get corresponding manifold points
    nearest_manifold_points = manifold_points[indices]
    
    # Compute weighted average
    mapped_points = torch.sum(nearest_manifold_points * kernel_weights.unsqueeze(-1), dim=1)
    
    return mapped_points, indices, kernel_weights



def _map_whole_trajectory2another_manifold_attention(
    traj_points, support_points_main, support_points_aux, sigma=0.1, device=None, aux_weight=1, distance_threshold=0.05
):
    """
    Attention-based projection mapping.
    
    traj_points: (N, Sample, dim_1)
    support_points_main: (n_support, dim_1)
    support_points_aux: (n_support, dim_2)
    
    Returns:
        mapped_indices: (N, Sample, n_support)
        weights: (N, Sample, n_support)
    """
    N, Sample, dim_1 = traj_points.shape
    n_support = support_points_main.shape[0]

    mapped_indices = []
    weights = []

    # Optional normalization for stability
    support_points_main_norm = F.normalize(support_points_main, dim=1)
    support_points_aux_norm = F.normalize(support_points_aux, dim=1)

    for t in range(N):
        traj_points_t = F.normalize(traj_points[t], dim=1)

        # Attention scores: dot product similarity
        attention_scores_main = torch.matmul(traj_points_t, support_points_main_norm.T)  # (Sample, n_support)

        if t == 0:
            # Apply Gaussian mask to enforce locality
            dists_main = torch.cdist(traj_points[t], support_points_main)
            mask = dists_main > 0.002
            attention_scores_main = attention_scores_main.masked_fill(mask, -1e6)

            # Soft selection with attention
            soft_weights = torch.softmax(attention_scores_main, dim=1)

            combined_weights = soft_weights

            mean_main = (support_points_main * combined_weights.unsqueeze(-1)).sum(dim=1)
            mean_aux = (support_points_aux * combined_weights.unsqueeze(-1)).sum(dim=1)

        else:
            # Compute attention scores between last mapped mean and support points
            mean_main_norm = F.normalize(mean_main, dim=1)
            mean_aux_norm = F.normalize(mean_aux, dim=1)

            attention_scores_main = torch.matmul(mean_main_norm, support_points_main_norm.T)
            attention_scores_aux = torch.matmul(mean_aux_norm, support_points_aux_norm.T)

            # Multi-space attention: weighted sum
            total_scores = attention_scores_main + aux_weight * attention_scores_aux

            dists_main = torch.cdist(traj_points[t], support_points_main)
            mask = dists_main > distance_threshold
            total_scores = total_scores.masked_fill(mask, -1e6)

            soft_weights = torch.softmax(total_scores, dim=1)

            combined_weights = soft_weights

            # Update mean positions
            mean_main = (support_points_main * combined_weights.unsqueeze(-1)).sum(dim=1)
            mean_aux = (support_points_aux * combined_weights.unsqueeze(-1)).sum(dim=1)

        indices = torch.arange(n_support, device=device).unsqueeze(0).repeat(Sample, 1)

        mapped_indices.append(indices)
        weights.append(combined_weights)

    mapped_indices = torch.stack(mapped_indices)
    weights = torch.stack(weights)

    return mapped_indices, weights


def _map_whole_trajectory2another_manifold_time_terminalMatching(
    traj_points,
    support_points_main,
    support_points_aux,
    k=10,
    alpha_main=100.0,
    sigma_aux=0.05
):
    """
    Multi-space KNN mapping with softmin weights and time continuity.

    Args:
        traj_points: (T, N, D)
        support_points_main: (M, D)
        support_points_aux: (M, D)
        k: number of neighbors
        alpha_main: softmin temperature for main space
        sigma_aux: bandwidth for auxiliary space

    Returns:
        indices: (T, N, k)
        joint_weights: (T, N, k)
    """
    traj_points = traj_points.to(dtype=support_points_main.dtype)
    T, N, D = traj_points.shape
    M = support_points_main.shape[0]

    traj_points_flat = traj_points.reshape(-1, D)

    # ============================
    # Step 1: path index (considering temporal continuity)
    # ============================
    prev_indices = None
    best_indices_list = []

    for t in range(T):
        traj_t = traj_points[t].reshape(N, D)

        # calculate distance between current time step and support
        dist_matrix_t = torch.cdist(traj_t, support_points_main)  # (N, M)
        distances_t, knn_indices_t = torch.topk(dist_matrix_t, k=k, dim=1, largest=False)  # (N, k)

        # main space Gaussian weights
        kernel_main = torch.exp(-distances_t ** 2 / (2 * 0.05 ** 2))
        kernel_main = kernel_main / (kernel_main.sum(dim=1, keepdim=True) + 1e-8)

        if t == 0:
            # first frame: directly take the point with the largest weight
            best_indices = knn_indices_t[torch.arange(N), torch.argmax(kernel_main, dim=1)]
            best_indices_list.append(best_indices)
            prev_indices = best_indices
            continue

        # temporal continuity auxiliary space weights
        prev_mapped_points = support_points_aux[prev_indices]  # (N, D)
        neighbor_points_aux = support_points_aux[knn_indices_t]  # (N, k, D)
        prev_expanded = prev_mapped_points.unsqueeze(1)  # (N, 1, D)
        distances_aux = torch.norm(neighbor_points_aux - prev_expanded, dim=-1)  # (N, k)

        kernel_aux = torch.exp(-distances_aux ** 2 / (2 * 0.025 ** 2))  # sigma_time fixed to 0.05
        kernel_aux = torch.clamp(kernel_aux, min=1e-4)

        joint_kernel = kernel_main * kernel_aux
        joint_kernel = joint_kernel / (joint_kernel.sum(dim=1, keepdim=True) + 1e-8)

        best_indices = knn_indices_t[torch.arange(N), torch.argmax(joint_kernel, dim=1)]
        best_indices_list.append(best_indices)

        prev_indices = best_indices

    # path index, continue batch processing
    path_indices = torch.stack(best_indices_list, dim=0)  # (T, N)

    # ============================
    # Step 2: efficient batch computation (batch processing)
    # ============================
    dist_matrix_support = torch.cdist(support_points_main, support_points_main)  # (M, M)
    distances_main_support, support_knn_indices = torch.topk(dist_matrix_support, k=k, dim=1, largest=False)

    mapped_support_indices = path_indices.view(T * N)
    neighbor_indices = support_knn_indices[mapped_support_indices]  # (T*N, k)

    neighbor_points_main = support_points_main[neighbor_indices]  # (T*N, k, D)
    neighbor_points_aux = support_points_aux[neighbor_indices]    # (T*N, k, D)

    traj_points_expanded = traj_points_flat.unsqueeze(1)  # (T*N, 1, D)

    distances_main = torch.norm(traj_points_expanded - neighbor_points_main, dim=-1)  # (T*N, k)

    # Softmin weights
    kernel_main = F.softmin(distances_main * alpha_main, dim=1)

    path_index_flat = path_indices.view(T * N)
    aux_path = support_points_aux[path_index_flat] # (T*N, D)
    aux_path_expanded = aux_path.unsqueeze(1) # (T*N, 1, D)
    #barycenter_aux = neighbor_points_aux.mean(dim=1, keepdim=True)  # (T*N, 1, D)
    distances_aux = torch.norm(neighbor_points_aux - aux_path_expanded, dim=-1)  # (T*N, k)

    kernel_aux = torch.exp(-distances_aux ** 2 / (2 * sigma_aux ** 2))
    kernel_aux = torch.clamp(kernel_aux, min=1e-4)

    joint_kernel = kernel_main * kernel_aux
    joint_kernel = joint_kernel / (joint_kernel.sum(dim=1, keepdim=True) + 1e-8)

    indices = neighbor_indices.view(T, N, k)
    joint_weights = joint_kernel.view(T, N, k)
    
    del dist_matrix_support, distances_main_support, support_knn_indices, mapped_support_indices, neighbor_indices, neighbor_points_main, neighbor_points_aux, traj_points_expanded, path_index_flat, aux_path, aux_path_expanded, distances_aux, kernel_aux, joint_kernel
    torch.cuda.empty_cache()

    return indices, joint_weights, path_indices

def _map_whole_trajectory2another_manifold_accrossSpace(
    traj_points,
    support_points_main,
    support_points_aux,
    k=10,
    alpha_main=100.0,   # kept for API compatibility; not used
    sigma_aux=0.05
):
    """
    Multi-space KNN mapping (time-flattened, no temporal continuity).

    Args:
        traj_points: (T, N, D) tensor
        support_points_main: (M, D) tensor
        support_points_aux: (M, D) tensor (one-to-one aligned with main)
        k: number of output neighbors per sample (use 20 if you want top-20)
        alpha_main: kept for API compatibility (not used)
        sigma_aux: bandwidth for Gaussian weights in BOTH spaces

    Returns:
        indices: (T, N, k) long tensor of selected support indices
        joint_weights: (T, N, k) float tensor of normalized weights
        path_indices: (T, N) long tensor of nearest (main-space) support per sample
    """
    # Do NOT use in-place .to_() etc.; make aligned copies (out-of-place)
    device = support_points_main.device
    dtype  = support_points_main.dtype
    traj_points = traj_points.to(device=device, dtype=dtype)
    support_points_aux = support_points_aux.to(device=device, dtype=dtype)

    T, N, D = traj_points.shape
    M = support_points_main.shape[0]
    TN = T * N

    # Flatten (T, N, D) -> (TN, D)
    X = traj_points.reshape(TN, D)

    # Candidate pool size from main space
    K_MAIN = min(50, M)
    K_OUT  = min(k, K_MAIN)

    # Preallocate outputs (these buffers don't require grad)
    out_indices = X.new_empty((TN, K_OUT), dtype=torch.long, device=device)
    out_weights = X.new_empty((TN, K_OUT), dtype=dtype, device=device)
    nearest_main_idx = X.new_empty((TN,), dtype=torch.long, device=device)

    # Chunk to control memory; keep everything out-of-place
    CHUNK = 4096 if TN >= 4096 else TN
    eps = torch.as_tensor(1e-12, dtype=dtype, device=device)
    two_sigma2 = torch.as_tensor(2.0 * (sigma_aux ** 2), dtype=dtype, device=device)

    for start in range(0, TN, CHUNK):
        end = min(start + CHUNK, TN)
        X_chunk = X[start:end]                         # (B, D)
        B = X_chunk.shape[0]

        # --- Main space distances & KNN (out-of-place) ---
        dist_main_full = torch.cdist(X_chunk, support_points_main)              # (B, M)
        dist_main_vals, dist_main_idx = torch.topk(dist_main_full, k=K_MAIN, dim=1, largest=False)  # (B, K_MAIN)

        # Anchor (nearest in main)
        anchor_idx = dist_main_idx[:, 0]                                        # (B,)
        nearest_main_idx[start:end] = anchor_idx

        # --- Aux distances between candidates and anchor's aux point (out-of-place) ---
        candidates_aux = support_points_aux[dist_main_idx]                       # (B, K_MAIN, D)
        anchor_aux = support_points_aux[anchor_idx].unsqueeze(1)                 # (B, 1, D)
        dist_aux_vals = torch.norm(candidates_aux - anchor_aux, dim=-1)          # (B, K_MAIN)

        # --- Gaussian weights in BOTH spaces (NO in-place clamp) ---
        w_main = torch.exp(-(dist_main_vals ** 2) / two_sigma2)                  # (B, K_MAIN)
        w_aux  = torch.exp(-(dist_aux_vals  ** 2) / two_sigma2)                  # (B, K_MAIN)

        # Multiply then normalize (out-of-place)
        w_joint = w_main * w_aux                                                 # (B, K_MAIN)
        w_sum = w_joint.sum(dim=1, keepdim=True)
        w_joint = w_joint / (w_sum + eps)

        # Top-K selection (indices are integer, not part of grad)
        topw, topi = torch.topk(w_joint, k=K_OUT, dim=1, largest=True)          # (B, K_OUT)
        top_indices = torch.gather(dist_main_idx, 1, topi)                       # (B, K_OUT)

        out_indices[start:end] = top_indices
        out_weights[start:end] = topw

        
        # Free temporaries early
        del dist_main_full, dist_main_vals, dist_main_idx, candidates_aux, anchor_aux, dist_aux_vals, w_main, w_aux, w_joint, topw, topi, top_indices
        torch.cuda.empty_cache() if X.is_cuda else None


    # Reshape to (T, N, ...)
    indices = out_indices.view(T, N, K_OUT)
    joint_weights = out_weights.view(T, N, K_OUT)
    path_indices = nearest_main_idx.view(T, N)

    return indices, joint_weights, path_indices


def _map_whole_trajectory2another_manifold_Simple(
    traj_points,
    support_points_main,
    support_points_aux,
    k=20,
    sigma=0.2
):
    """
    Simplest variant: time-flattened KNN in MAIN space only.
    Weights come from MAIN-space Gaussian; AUX is only used for mapping by indices.

    Args:
        traj_points: (T, N, D) tensor
        support_points_main: (M, D) tensor
        support_points_aux: (M, D) tensor (one-to-one with main)
        k: number of neighbors to return
        alpha_main: kept for API compatibility (not used)
        sigma_aux: Gaussian bandwidth for MAIN-space distances

    Returns:
        indices: (T, N, k) long tensor of selected support indices (by MAIN KNN)
        joint_weights: (T, N, k) float tensor of normalized MAIN Gaussian weights
        path_indices: (T, N) long tensor of nearest MAIN support per sample
    """
    # Device / dtype alignment (out-of-place)
    device = support_points_main.device
    dtype  = support_points_main.dtype
    traj_points = traj_points.to(device=device, dtype=dtype)
    support_points_aux = support_points_aux.to(device=device, dtype=dtype)

    T, N, D = traj_points.shape
    M = support_points_main.shape[0]
    TN = T * N

    # Flatten time
    X = traj_points.reshape(TN, D)

    # Effective k
    K = min(int(k), M)
    if K <= 0:
        raise ValueError("k must be >= 1 and <= number of support points.")

    # Outputs (do not require grad)
    out_indices = torch.empty((TN, K), dtype=torch.long, device=device)
    out_weights = torch.empty((TN, K), dtype=dtype, device=device)
    nearest_idx = torch.empty((TN,), dtype=torch.long, device=device)

    # Chunking to control memory
    CHUNK = 4096 if TN >= 4096 else TN
    eps = torch.as_tensor(1e-12, dtype=dtype, device=device)
    two_sigma2 = torch.as_tensor(2.0 * (sigma ** 2), dtype=dtype, device=device)

    for start in range(0, TN, CHUNK):
        end = min(start + CHUNK, TN)
        Xc = X[start:end]  # (B, D)

        # MAIN-space distances and KNN
        dist_full = torch.cdist(Xc, support_points_main)                     # (B, M)
        dists, idxs = torch.topk(dist_full, k=K, dim=1, largest=False)       # (B, K)

        # Nearest (for path_indices)
        nearest_idx[start:end] = idxs[:, 0]

        # Gaussian weights in MAIN space (no in-place)
        w = torch.exp(-(dists ** 2) / two_sigma2)                            # (B, K)
        wsum = w.sum(dim=1, keepdim=True)
        w = w / (wsum + eps)

        out_indices[start:end] = idxs
        out_weights[start:end] = w


    # Reshape back to (T, N, K)
    indices = out_indices.view(T, N, K)
    joint_weights = out_weights.view(T, N, K)
    path_indices = nearest_idx.view(T, N)

    return indices, joint_weights, path_indices

def knn_aux_mean_distance_loss(traj, support_main, support_aux, k=20, sigma=1.0):
    """
    traj: (N, Sample, dim)
    support_main: (n_support, dim)
    support_aux: (n_support, dim)
    k: number of nearest neighbors
    sigma: softmax temperature
    alpha: consistency loss weight
    Returns:
        loss: scalar
    """
    N, Sample, dim = traj.shape

    # Reshape traj to (N*Sample, dim)
    traj_flat = traj.reshape(-1, dim)  # (B, dim), B = N*Sample

    # Compute pairwise distances in main space: (B, n_support)
    dists_main = torch.cdist(traj_flat, support_main)

    # Find top-k neighbors (B, k)
    knn_dists, knn_indices = torch.topk(-dists_main, k=k, dim=1)
    knn_dists = -knn_dists

    # Softmax weights: (B, k)
    weights = torch.softmax(-knn_dists / sigma, dim=1)

    # Gather aux neighbors: (B, k, dim)
    aux_knn = support_aux[knn_indices]

    # Compute weighted mean in aux space: (B, dim)
    weights_expanded = weights.unsqueeze(-1)  # (B, k, 1)
    mean_aux_soft = torch.sum(aux_knn * weights_expanded, dim=1)

    # Variance loss: compute distances to weighted center
    aux_diff = aux_knn - mean_aux_soft[:, None, :]  # (B, k, dim)
    aux_dist = torch.norm(aux_diff, dim=-1, p=2)  # (B, k)
   

    '''
    # Consistency loss: adjacent time steps
    mean_aux_soft_time = mean_aux_soft.view(N, Sample, dim)  # (N, Sample, dim)

    # Shifted difference: (N, Sample-1, dim)
    consistency_diff = mean_aux_soft_time[:, 1:, :] - mean_aux_soft_time[:, :-1, :]
    loss_consistency = torch.norm(consistency_diff, dim=-1).mean()
    '''
    # Total loss
    threshold = 0.1
    sharpness = 50
    
    aux_smooth = 1/(1 + torch.exp(-sharpness * (aux_dist - threshold)))
    
    loss = aux_smooth.mean()

    return loss


def gaussian_kernel(window_size=5, std=1.0, device=torch.device("cuda:0")):
    center = window_size // 2
    x = torch.arange(window_size, device=device) - center
    kernel = torch.exp(-0.5 * (x / std)**2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1)  # shape (1, 1, window_size)

def smooth_trajectory(trajectory, window_size=5):
    """
    smooth the trajectory (keep the first and last frame unchanged), support efficient vectorization.
    Args:
        trajectory: (N_time, Sample, dim)
        window_size: odd, window size
    Returns:
        smoothed: (N_time, Sample, dim)
    """
    assert window_size % 2 == 1, "window_size must be odd."
    N_time, Sample, dim = trajectory.shape
    half_w = window_size // 2

    # 1. reshape: (Sample * dim, N_time)
    traj = trajectory.permute(1, 2, 0).reshape(Sample * dim, N_time).unsqueeze(1)  # (Batch, 1, Time)

    # 2. padding: keep boundary
    traj_padded = F.pad(traj, pad=(half_w, half_w), mode='replicate')  # (Batch, 1, Time + pad)

    # 3. kernel: same kernel for all batches
    #kernel = torch.ones(1, 1, window_size, device=trajectory.device) / window_size
    kernel = gaussian_kernel(window_size=window_size, std=2.0, device=trajectory.device)
    # 4. apply same kernel to each sequence
    smoothed = F.conv1d(traj_padded, kernel, groups=1)  # (Batch, 1, Time)

    # 5. reshape back
    smoothed = smoothed.squeeze(1).reshape(Sample, dim, N_time).permute(2, 0, 1).contiguous()  # (N_time, Sample, dim)

    smoothed[0] = trajectory[0]
    smoothed[-1] = trajectory[-1]

    return smoothed


import numpy as np

def project_trajectory(secondary_trajectory, total_secondary, threshold=1e-2):
    """
    PyTorch version: Project secondary_trajectory to (D+1) dim, last dim is distance to manifold (tanh mapped).

    Args:
        secondary_trajectory: (T, N, D) torch tensor (can be on GPU)
        total_secondary: (N, D) torch tensor (can be on GPU)
        threshold: distance threshold

    Returns:
        projected_trajectory: (T, N, D+1) torch tensor
    """
    T, N, D = secondary_trajectory.shape

    # Expand dimensions for broadcasting
    traj_expanded = secondary_trajectory.unsqueeze(2)  # (T, N, 1, D)
    manifold_expanded = total_secondary.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

    # Compute pairwise distances: (T, N, N)
    distances = torch.norm(traj_expanded - manifold_expanded, dim=-1)

    # Get min distance to the manifold for each point: (T, N)
    min_distances, _ = distances.min(dim=-1)

    # Apply threshold and tanh mapping
    distance_values = torch.where(min_distances < threshold, torch.zeros_like(min_distances), torch.tanh(min_distances * 20)) * 1000

    # Concatenate the distance as the (D+1)-th dimension
    projected_trajectory = torch.cat([secondary_trajectory, distance_values.unsqueeze(-1)], dim=-1)

    return projected_trajectory


# ============================================================
# Paired (primary, secondary) trajectory rollout from initial cells
#
# For inference / visualization: given a trained primary velocity field
# `func` and a frozen primary->secondary map `T_model`, roll out the
# primary ODE from a fixed set of initial cells on a dt-based fine time
# grid that follows the training time-point schedule, then apply T_model
# at every time slice to obtain the paired secondary trajectory.
#
# The time grid is built segment-by-segment to mirror the convention
# used in training (Training.py::_build_time_steps_list), so the
# returned trajectory aligns with how the model was trained: each
# segment [t_i, t_{i+1}] is discretized with step `dt`, and consecutive
# segments share their endpoint (deduplicated in the concatenated grid).
# ============================================================

def _build_dt_time_grid(time_points, dt, dtype, device):
    """
    Build a fine time grid from a list of milestone time points.

    For each consecutive pair (t_lo, t_hi), use n = int((t_hi - t_lo) / dt)
    + 1 points (so the spacing is at most dt and t_hi is hit exactly).
    Concatenate segments while dropping the duplicated boundary point.
    """
    grid = [float(time_points[0])]
    for i in range(len(time_points) - 1):
        t_lo = float(time_points[i])
        t_hi = float(time_points[i + 1])
        n_steps = int((t_hi - t_lo) / dt) + 1
        seg = torch.linspace(t_lo, t_hi, n_steps).tolist()
        grid.extend(seg[1:])   # skip seg[0] (= prev segment's last point)
    return torch.tensor(grid, dtype=dtype, device=device)


def generate_paired_trajectories(
    func,
    initial_primary,
    time_points,
    T_model=None,
    dt=0.1,
    unbalancedModel=False,
    method='rk4',
    device=None,
):
    """
    Roll out a primary trajectory; optionally also a paired secondary
    trajectory via T_model.

    The function always returns 4 items so the call signature is stable;
    `secondary_traj` is None when T_model is not supplied (single-space
    use case, e.g. RNA-only inference). Callers can simply unpack
    and ignore the unused secondary slot.

    Args
    ----
        func             : trained primary velocity field. Must follow the
                           same `forward(t, states)` signature used in
                           training (Neural.MLPVectorField). Both balanced
                           and unbalanced variants are supported via
                           `unbalancedModel`.
        initial_primary  : (N, d_primary) tensor of initial cells in primary
                           space, typically the t=t_start cells AFTER the
                           same normalization used in training.
        time_points      : list[float] of milestone times (e.g.
                           [0, 1, 2, 2.5] for Gastrulation; [0, 1, 1.5, 2]
                           for MouseBrain). The first / last entries are
                           used as t_start / t_end.
        T_model          : optional. Frozen primary->secondary map. Must
                           satisfy the duck-typed contract
                           T_model(q, t) -> y with q of shape (..., d_p),
                           t scalar or (...,) broadcastable to q's leading
                           shape, and output (..., d_s). If None, only
                           the primary trajectory is generated.
        dt               : fine integration step size within each segment
                           (default 0.1, matching args.dt convention).
        unbalancedModel  : if True, integrate with growth state alongside
                           position to match the unbalanced training mode.
        method           : torchdiffeq ODE solver. Default 'rk4' (fixed-
                           step, deterministic, more accurate than the
                           'euler' used in training).
        device           : optional override; defaults to
                           ``initial_primary.device``.

    Returns
    -------
        primary_traj   : (T, N, d_primary)
        secondary_traj : (T, N, d_secondary) if T_model else None
        t_grid         : (T,) time at each slice
        extras         : dict — unbalanced: {'lnw', 'e_v', 'e_g'};
                                balanced  : {'e'}
    """
    if device is None:
        device = initial_primary.device

    # ---- 1. Build the dt-based fine time grid ----
    t_grid = _build_dt_time_grid(
        time_points, dt,
        dtype=initial_primary.dtype, device=device,
    )

    # ---- 2. Primary ODE rollout from initial_primary ----
    N = initial_primary.shape[0]
    if unbalancedModel:
        lnw_0 = torch.log(torch.ones(N, 1, device=device) / N)
        e_v_0 = torch.zeros(N, 1, device=device)
        e_g_0 = torch.zeros(N, 1, device=device)
        with torch.no_grad():
            primary_traj, lnw_t, e_v_t, e_g_t = odeint(
                func,
                (initial_primary, lnw_0, e_v_0, e_g_0),
                t_grid,
                method=method,
            )
        extras = {'lnw': lnw_t, 'e_v': e_v_t, 'e_g': e_g_t}
    else:
        e_0 = torch.zeros(N, 1, device=device)
        with torch.no_grad():
            primary_traj, e_t = odeint(
                func,
                (initial_primary, e_0),
                t_grid,
                method=method,
            )
        extras = {'e': e_t}

    # ---- 3. Optionally apply T_model to get the secondary trajectory ----
    if T_model is None:
        return primary_traj, None, t_grid, extras

    # Shape contract follows TrainLoss.short_term_loss's T_model call:
    # q is (T, N, d_p) and t is broadcast to (T, N).
    T_steps = primary_traj.shape[0]
    t_q = (t_grid.to(primary_traj.dtype)
                 .reshape(T_steps, 1)
                 .expand(T_steps, N))
    with torch.no_grad():
        secondary_traj = T_model(primary_traj, t_q)

    return primary_traj, secondary_traj, t_grid, extras