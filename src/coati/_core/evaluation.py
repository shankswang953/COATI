"""
Evaluation pipeline for trained SyncOT checkpoints.

Batch-evaluate a list of checkpoints (or sweep configs) and return all
metrics as a list of dicts, ready to be converted to a pandas DataFrame.

The cross-modal mapping model (AlignMLP or otherwise) is supplied by the
caller so that architecture changes don't require touching src/.
"""

import os
import numpy as np
import torch
from geomloss import SamplesLoss

from .DataLoad import (
    load_source_data,
    normalize_to_unit_cube_global,
    get_batch_by_index,
)
from .Neural import MLPVectorField
from .TrainLoss import calculate_density_loss
from .utility import forward_ode, load_checkpoint


# ============================================================
# Shared resource loader (call once, reuse across evaluations)
# ============================================================
def setup_evaluation(
    args,
    map_model,                                        # user-provided, already loaded + on device + .eval()
    primal_norm_path='../data/primal_norm_params.pt',
    sec_norm_path='../data/secondary_norm_params.pt',
    w2_blur=0.1,
    device=None,
):
    """
    Load data and W2 solver once. Reuse across many evaluations.

    Args:
        args: Args object (for data paths, time_points, etc.).
        map_model: primary->secondary mapping module built by the caller.
            Should already be loaded, moved to device, set to eval().
        primal_norm_path / sec_norm_path: normalization params on disk.
        w2_blur: blur parameter for the Sinkhorn W2 solver.
        device: optional override; if None, resolved from args.

    Returns:
        dict with keys: device, primary_data, secondary_data,
            total_primal, total_secondary, map_model, w2_loss.
    """
    if device is None:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"[eval-setup] device = {device}")

    # ---- Primary data ----
    primary_data = load_source_data(args.data_path, args.time_labels, args.support_points, DataShape=False)
    primary_data = [d.to(device) for d in primary_data]

    primal_norm_params = torch.load(primal_norm_path)
    primary_data, _ = normalize_to_unit_cube_global(
        primary_data, args.time_points, norm_params=primal_norm_params
    )
    total_primal = torch.cat(primary_data, dim=0)

    # ---- Secondary data ----
    secondary_data = load_source_data(args.sync_data_path, args.time_labels, args.support_points, DataShape=False)
    secondary_data = [d.to(device) for d in secondary_data]

    sec_norm_params = torch.load(sec_norm_path)
    secondary_data, _ = normalize_to_unit_cube_global(
        secondary_data, args.time_points, norm_params=sec_norm_params
    )
    total_secondary = torch.cat(secondary_data, dim=0)

    # ---- W2 loss object ----
    w2_loss = SamplesLoss(loss="sinkhorn", p=2, blur=w2_blur, backend="tensorized").to(device)

    return {
        'device': device,
        'primary_data': primary_data,
        'secondary_data': secondary_data,
        'total_primal': total_primal,
        'total_secondary': total_secondary,
        'map_model': map_model,
        'w2_loss': w2_loss,
    }


# ============================================================
# Checkpoint filename builder (mirrors src/utility.py:build_exp_name)
# ============================================================
def build_ckpt_name(args, iter_num=20000):
    """Reconstruct checkpoint filename from args."""
    parts = [
        f"s{args.seed}",
        f"e{args.energy_coefficient}",
        f"m{args.pdf_coefficient}",
        f"d{args.density_coefficient}",
    ]
    if getattr(args, 'sync_loss', False):
        parts.append(f"a{args.sync_weight}")
    parts.append(f"iter{iter_num}")
    return "ckpt_" + "_".join(parts) + ".pth"


# ============================================================
# Per-checkpoint evaluation
# ============================================================
def evaluate_checkpoint(ckpt_filename, args, shared):
    """
    Load one checkpoint and compute all primal / secondary metrics.

    Args:
        ckpt_filename: e.g. 'ckpt_s0_e0.1_m1.0_d0.01_a0.5_iter20000.pth'.
            Must be relative to args.train_dir.
        args: Args object (hyperparameters for this experiment).
        shared: dict returned by setup_evaluation.

    Returns:
        dict of metrics (floats).
    """
    device = shared['device']
    primary_data = shared['primary_data']
    secondary_data = shared['secondary_data']
    total_secondary = shared['total_secondary']
    map_model = shared['map_model']
    w2_loss = shared['w2_loss']

    # ---- Build model & load weights ----
    func = MLPVectorField(
        dim=args.otdim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        activation=args.activation,
        unbalanced=args.unbalancedModel,
        alpha_growth=args.alpha_growth,
    ).to(device)
    load_checkpoint(func, args, ckpt_path=ckpt_filename)
    func.eval()

    # ---- Forward ODE ----
    n_Day1 = primary_data[0].shape[0]
    chosen_indices = np.arange(0, n_Day1, 1)
    initial_distribution = get_batch_by_index(chosen_indices, device, primary_data[0])[0]

    dt = args.dt
    start = args.time_points[0]
    terminal = args.time_points[-1]
    viz_timesteps = round((terminal - start) / dt) + 1

    with torch.no_grad():
        if args.unbalancedModel:
            all_samples_forward, lnw_t, e_v_t, e_g_t = forward_ode(
                func, initial_distribution, device, args.unbalancedModel,
                viz_timesteps=viz_timesteps, t_start=start, t_end=terminal,
            )
            e_t = None
        else:
            all_samples_forward, e_t = forward_ode(
                func, initial_distribution, device, args.unbalancedModel,
                viz_timesteps=viz_timesteps, t_start=start, t_end=terminal,
            )
            e_v_t = e_g_t = None

    # ---- Compute trajectory indices for each reference time ----
    # The trajectory is generated at viz_timesteps points uniformly spaced
    # between `start` and `terminal`. The reference data, however, lives at
    # args.time_points (which are NOT necessarily uniform).
    # For each reference time t_ref, we map it to the nearest trajectory
    # index via:  idx = round((t_ref - start) / dt).
    # We skip the first reference time because trajectory[0] is exactly the
    # initial distribution by construction (always matches primary_data[0]).
    ref_times = list(args.time_points)
    ref_indices = []
    for i, t_ref in enumerate(ref_times):
        idx = int(round((t_ref - start) / dt))
        idx = max(0, min(idx, viz_timesteps - 1))   # clamp for safety
        ref_indices.append(idx)

    # Map to secondary space ahead of time so we can compute W2 at every ref.
    # Pass per-step time alongside the trajectory so T models with explicit
    # time conditioning (FiLMMLP / TemporalMLP) receive the same (q, t)
    # signature they saw during training (see TrainLoss.short_term_loss).
    # PlainMLP's forward accepts (q, t) and silently ignores t, so this is
    # also safe for the time-agnostic case. Models that follow the project's
    # T convention preserve leading dimensions on (..., d_in) -> (..., d_out),
    # so no reshape is needed afterwards.
    T, N, D = all_samples_forward.shape
    with torch.no_grad():
        t_per_step = torch.linspace(start, terminal, T,
                                    device=all_samples_forward.device,
                                    dtype=all_samples_forward.dtype)
        t_q = t_per_step.reshape(T, 1).expand(T, N)   # (T, N)
        sec_traj = map_model(all_samples_forward, t_q)

    # ---- Per-reference W2 (skip first ref = initial distribution) ----
    w2_primal_per_ref = []
    w2_secondary_per_ref = []
    for i in range(1, len(ref_times)):
        idx = ref_indices[i]
        w2_p = w2_loss(all_samples_forward[idx], primary_data[i])
        w2_s = w2_loss(sec_traj[idx], secondary_data[i].to(device))
        w2_primal_per_ref.append(w2_p.item())
        w2_secondary_per_ref.append(w2_s.item())

    w2_primal_sum    = float(sum(w2_primal_per_ref))
    w2_secondary_sum = float(sum(w2_secondary_per_ref))
    w2_primal_terminal    = w2_primal_per_ref[-1]    if w2_primal_per_ref    else 0.0
    w2_secondary_terminal = w2_secondary_per_ref[-1] if w2_secondary_per_ref else 0.0

    # ---- Energy & density (unchanged: still computed across the full traj / terminal) ----
    # Transportation energy = 0.5 * sum ||v_t||^2 * dt
    pairs = [
        (all_samples_forward[i], all_samples_forward[i + 1])
        for i in range(len(all_samples_forward) - 1)
    ]
    vel = [torch.norm((p2 - p1) / dt, p=2, dim=1) for p1, p2 in pairs]
    transportation_energy = torch.sum(torch.stack(vel) ** 2 * dt, dim=0).mean() * 0.5

    if args.unbalancedModel:
        neural_energy_vel = e_v_t[-1].sum().item()
        neural_energy_growth = e_g_t[-1].sum().item()
        neural_energy = neural_energy_vel + neural_energy_growth
    else:
        neural_energy = e_t[-1].mean().item()
        neural_energy_vel = None
        neural_energy_growth = None

    density_primal = calculate_density_loss(
        all_samples_forward[-1], primary_data[-1].to(device),
        k=30, hinge_value=0.01,
    )

    pairs_sec = [(sec_traj[i], sec_traj[i + 1]) for i in range(len(sec_traj) - 1)]
    vel_sec = [torch.norm((p2 - p1), p=2, dim=1) for p1, p2 in pairs_sec]
    secondary_energy = (torch.stack(vel_sec) ** 2 / dt).sum(dim=0).mean() * 0.5

    density_secondary = calculate_density_loss(
        sec_traj[-1], secondary_data[-1].to(device),
        k=30, hinge_value=0.01,
    )

    # ---- Aggregate losses (matching training loss structure) ----
    # NOTE: total_primal_loss now uses the *sum* W2 across all reference time
    # points (was previously terminal-only). This is a more comprehensive
    # measure of distribution match along the whole trajectory.
    total_primal_loss = (
        w2_primal_sum * args.pdf_coefficient
        + transportation_energy.item() * args.energy_coefficient
        + density_primal.item() * args.density_coefficient
    )
    energy_density_primal = (
        transportation_energy.item() * args.energy_coefficient
        + density_primal.item() * args.density_coefficient
    )
    total_secondary_loss = (
        secondary_energy.item() * args.energy_coefficient
        + density_secondary.item() * args.density_coefficient
    )

    # Per-time-named columns for easy DataFrame analysis
    # (e.g. 'w2_primal_t1.0' = W2 at the second ref time, etc.)
    w2_primal_named    = {f'w2_primal_t{ref_times[i]}':    w2_primal_per_ref[i - 1]
                         for i in range(1, len(ref_times))}
    w2_secondary_named = {f'w2_secondary_t{ref_times[i]}': w2_secondary_per_ref[i - 1]
                         for i in range(1, len(ref_times))}

    return {
        'ckpt': ckpt_filename,
        # ---- Primary W2: list, sum, terminal, plus per-time named columns ----
        'w2_primal_list':     w2_primal_per_ref,         # [W2_t1, W2_t2, ...]
        'w2_primal':          w2_primal_sum,             # sum across ref times
        'w2_primal_terminal': w2_primal_terminal,        # last (== old w2_primal)
        **w2_primal_named,                               # w2_primal_t1.0, ...
        # ---- Other primary metrics ----
        'transportation_energy_primal': transportation_energy.item(),
        'neural_energy_primal': neural_energy,
        'neural_energy_vel': neural_energy_vel,
        'neural_energy_growth': neural_energy_growth,
        'density_primal': density_primal.item(),
        # ---- Secondary W2: list, sum, terminal, plus per-time named columns ----
        'w2_secondary_list':     w2_secondary_per_ref,
        'w2_secondary':          w2_secondary_sum,
        'w2_secondary_terminal': w2_secondary_terminal,
        **w2_secondary_named,
        # ---- Other secondary metrics ----
        'energy_secondary': secondary_energy.item(),
        'density_secondary': density_secondary.item(),
        # ---- Aggregated ----
        'total_primal_loss': total_primal_loss,
        'energy_density_primal': energy_density_primal,
        'total_secondary_loss': total_secondary_loss,
    }


# ============================================================
# Batch evaluation over a list of configs
# ============================================================
def evaluate_configs(
    configs,
    args_factory,
    map_model,
    iter_num=20000,
    primal_norm_path='../data/primal_norm_params.pt',
    sec_norm_path='../data/secondary_norm_params.pt',
    verbose=True,
):
    """
    Evaluate all checkpoints for a list of sweep configs.

    Args:
        configs: list of dicts, e.g. EXPERIMENTS from sweep_configs.py.
        args_factory: zero-arg callable returning a fresh Args instance
            (usually just the ``Args`` class).
        map_model: primary->secondary mapping module, prepared by the caller.
        iter_num: which iteration's checkpoint to load (default 20000).

    Returns:
        list of dict: one entry per config, with both config fields and
            all metrics. Missing checkpoints produce an entry with
            `'error': 'not_found'` so the batch doesn't crash.
    """
    args_shared = args_factory()
    shared = setup_evaluation(
        args_shared,
        map_model=map_model,
        primal_norm_path=primal_norm_path,
        sec_norm_path=sec_norm_path,
    )

    results = []
    for i, cfg in enumerate(configs):
        args = args_factory()
        for k, v in cfg.items():
            setattr(args, k, v)

        ckpt_name = build_ckpt_name(args, iter_num=iter_num)

        if verbose:
            print(f"[{i+1}/{len(configs)}] {ckpt_name}", flush=True)

        ckpt_path = os.path.join(args.train_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            results.append({**cfg, 'ckpt': ckpt_name, 'iter': iter_num, 'error': 'not_found'})
            if verbose:
                print("  -> not found, skipping")
            continue

        try:
            metrics = evaluate_checkpoint(ckpt_name, args, shared)
            results.append({**cfg, 'iter': iter_num, **metrics})
        except Exception as e:
            if verbose:
                print(f"  -> ERROR: {e}")
            results.append({**cfg, 'ckpt': ckpt_name, 'iter': iter_num, 'error': str(e)})

    return results