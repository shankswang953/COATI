"""
Training pipeline for SyncOT neural ODE trajectory inference.

This module wraps the full training workflow (data loading, normalization,
model / optimizer / scheduler construction, checkpoint warm-start, and the
training loop) behind a single entry point so that experiment scripts
can stay short and declarative.

Typical usage
-------------
    from Args import Args
    from .Training import run_training

    args = Args()

    map_model = None
    if args.sync_loss:
        map_model = build_map_model(args)     # user-defined, e.g. AlignMLP
        map_model.load_state_dict(...)
        for p in map_model.parameters():
            p.requires_grad = False
        map_model.eval()

    run_training(
        args,
        map_model=map_model,
        warm_start_path=None,                  # or 'ckpt_xxx.pth' to resume
    )

Single-space vs multi-space
---------------------------
- ``args.sync_loss = False``  : only primary data is loaded / used.
                                ``map_model`` and secondary_data are ignored.
- ``args.sync_loss = True``   : ``map_model`` is REQUIRED.
                                Secondary data is loaded from
                                ``args.sync_data_path`` and normalized with
                                ``sec_norm_path``.
"""

import os
import torch
import torch.optim as optim

from .DataLoad import load_source_data, normalize_to_unit_cube_global
from .Neural import MLPVectorField, RunningAverageMeter
from .utility import set_global_seed, load_checkpoint
from .Epoch import train_path_epoch


# NOTE: src/ is NN-agnostic by design. The caller (run.py / training
# driver script) is responsible for instantiating, loading weights into,
# freezing, and moving any external T model to the correct device, then
# passing it to run_training(..., T_model=T_model). No path/class names
# or checkpoint-format assumptions live in src/.


def _resolve_device(args):
    """Pick CUDA / MPS / CPU following the same rules as the run script."""
    if getattr(args, "device", "auto") != "auto":
        return torch.device(args.device)
    if torch.cuda.is_available() and args.gpu >= 0:
        return torch.device(f'cuda:{args.gpu}')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _build_time_steps_list(time_points, dt, device):
    """Build per-segment (backward, forward) time-step tensors for short-term ODE integration."""
    time_steps_list = []
    for t_idx in range(len(time_points) - 1):
        t0, t1 = time_points[t_idx], time_points[t_idx + 1]
        n_steps = int(round((t1 - t0) / dt)) + 1
        time_steps_short_backward = torch.linspace(t1, t0, n_steps).to(device)
        time_steps_short_forward  = torch.linspace(t0, t1, n_steps).to(device)
        time_steps_list.append((time_steps_short_backward, time_steps_short_forward))
    return time_steps_list


def run_training(
    args,
    map_model=None,
    T_model=None,
    primal_norm_path=None,
    sec_norm_path=None,
    warm_start_path=None,
    metric_primal=None,
    metric_secondary=None,
):
    """
    Run the full training pipeline.

    Args:
        args: Args / argparse.Namespace with fields used throughout the pipeline
            (seed, gpu, data_path, time_points, dt, hidden_dim, n_layers, otdim,
            lr, niters, sync_loss, sync_data_path, ...).
        map_model: RNA->ATAC (primary->secondary) mapping module.
            Required when ``args.sync_loss=True``; ignored otherwise.
            The caller is responsible for loading weights, moving to device,
            freezing parameters, and setting ``.eval()``.
        primal_norm_path: path to the primary-space normalization params (.pt).
            Pass ``None`` to skip primal normalization (use raw data as-is).
        sec_norm_path: path to the secondary-space normalization params (.pt).
            Only used when ``args.sync_loss=True``. Pass ``None`` to skip
            secondary normalization (use raw data as-is).
        warm_start_path: optional checkpoint filename (relative to
            ``args.train_dir``). If given, ``func`` is initialized from this
            checkpoint before training (warm-start). Pass None for a fresh run.
        metric_primal / metric_secondary: optional Riemannian metrics forwarded
            to ``train_path_epoch``.

    Returns:
        dict with keys:
            'func'      : trained MLPVectorField
            'optimizer' : final optimizer state
            'scheduler' : final scheduler state
            'results'   : whatever train_path_epoch returns
    """
    # ---------- Reproducibility ----------
    set_global_seed(args.seed)

    # ---------- Device ----------
    device = _resolve_device(args)
    print(f"[Training] device = {device}")
    print(f"[Training] sync_loss = {args.sync_loss}")

    # ---------- Sanity check for multi-space training ----------
    # src is fully NN-agnostic. Dispatch priority:
    #   T_model passed   -> use that frozen external NN (architecture
    #                       is entirely the caller's choice; src only
    #                       calls model(q, t)).
    #   map_model passed -> legacy CrossModalVAE-style mapping.
    #   neither          -> non-parametric kernel projection, with the
    #                       sub-variant chosen by args.map_type
    #                       ('kernel' = scalar sigma; 'accurate_kernel'
    #                       = per-cell adaptive sigma).
    if args.sync_loss:
        map_type = getattr(args, 'map_type', None)
        if T_model is not None:
            print(f"[Training] using caller-provided T_model "
                  f"(frozen external NN; map_type={map_type!r}).")
            args.T_model = T_model
            map_model = None
        elif map_model is not None:
            print(f"[Training] using caller-provided map_model "
                  f"(legacy CrossModalVAE-style; map_type={map_type!r}).")
        elif map_type == 'kernel':
            print("[Training] map_type=kernel — non-parametric Gaussian "
                  "kernel projection with a single global sigma.")
        elif map_type == 'accurate_kernel':
            print("[Training] map_type=accurate_kernel — non-parametric "
                  "Gaussian kernel projection with per-cell adaptive sigma "
                  "(k-NN distance, globally percentile-clipped).")
        else:
            raise ValueError(
                f"sync_loss=True but the caller passed neither T_model "
                f"nor map_model, and args.map_type={map_type!r} is not "
                f"a kernel-mode label ('kernel' / 'accurate_kernel'). "
                f"Either pass T_model=<any frozen nn.Module with "
                f"forward(q, t)>, pass map_model=<CrossModalVAE-style>, "
                f"or set args.map_type to a kernel mode."
            )

    # ---------- Primary data ----------
    primary_data = load_source_data(args.data_path, args.time_labels, args.support_points)
    primary_data = [d.to(device) for d in primary_data]

    if primal_norm_path is not None:
        primal_norm_params = torch.load(primal_norm_path)
        primary_data, _ = normalize_to_unit_cube_global(
            primary_data, args.time_points, norm_params=primal_norm_params
        )
        print(f"[Training] primal data normalized via {primal_norm_path}")
    else:
        print("[Training] primal_norm_path=None, using raw primal data (no normalization)")

    # ---------- Secondary data (only when sync_loss=True) ----------
    secondary_data = None
    if args.sync_loss:
        secondary_data = load_source_data(args.sync_data_path, args.time_labels, args.support_points)
        secondary_data = [d.to(device) for d in secondary_data]

        if sec_norm_path is not None:
            sec_norm_params = torch.load(sec_norm_path)
            secondary_data, _ = normalize_to_unit_cube_global(
                secondary_data, args.time_points, norm_params=sec_norm_params
            )
            print(f"[Training] secondary data normalized via {sec_norm_path}")
        else:
            print("[Training] sec_norm_path=None, using raw secondary data (no normalization)")

    # ---------- Per-time medians + adaptive Sinkhorn blur ----------
    # Compute median pairwise distance per timepoint once for primary,
    # derive both kernel sigma (if applicable) and per-time Sinkhorn blur
    # from this single computation. Per-time, per-modality blur is essential
    # because different distributions have different intrinsic length scales:
    # a single fixed blur cannot be appropriate for both modalities and all
    # timepoints. Median heuristic (blur = c * median pairwise distance) is
    # the field convention; floor at 1e-6 to guard against degenerate
    # distributions.
    from .MapSpace import (compute_per_time_median_distance,
                              compute_noise_floor_per_time,
                              compute_per_cell_sigma_per_time,
                              clip_sigma_globally)
    from geomloss import SamplesLoss

    # ---- Defaults for diagnostics-related args ---------------------------
    # Set sensible defaults for parameters that the user may not have added
    # to Args.py. These cover (a) the sample size for median pairwise
    # distance estimation (used to compute Sinkhorn blur and kernel sigma)
    # and (b) the sample size for noise-floor estimation. n_repeats for
    # the floor is hardcoded inside compute_noise_floor_per_time (50).
    if not hasattr(args, 'median_n_sample'):
        args.median_n_sample = 2000
        print(f"[Training] args.median_n_sample not set, default = "
              f"{args.median_n_sample}")
    if not hasattr(args, 'floor_n_sample'):
        args.floor_n_sample = args.num_samples
        print(f"[Training] args.floor_n_sample not set, default = "
              f"args.num_samples = {args.floor_n_sample}")

    blur_coeff = getattr(args, 'blur_coeff', 0.05)

    medians_primary = compute_per_time_median_distance(
        primary_data, n_sample=args.median_n_sample
    )
    args.median_dist_primary = medians_primary
    args.blur_per_time_primary = [max(blur_coeff * m, 1e-6) for m in medians_primary]
    print(f"[Training] primary median per time = "
          f"{[f'{m:.4f}' for m in medians_primary]}")
    print(f"[Training] primary blur   per time = "
          f"{[f'{b:.4e}' for b in args.blur_per_time_primary]}")

    # Kernel sigma dispatch by map_type. Three paths:
    #   'kernel'          -> single global scalar sigma (reuses medians).
    #   'accurate_kernel' -> per-cell adaptive sigma, k-NN based,
    #                        globally percentile-clipped (see Section 7.6).
    #   anything else (incl. 'MLP') -> kernel projection not used.
    args.kernel_sigma = None
    args.sigma_per_cell_primary = None
    if args.sync_loss:
        _map_type = getattr(args, 'map_type', 'MLP')
        if _map_type == 'kernel':
            args.kernel_sigma = sum(medians_primary) / len(medians_primary)
            print(f"[Training] auto-estimated kernel sigma = "
                  f"{args.kernel_sigma:.4f} "
                  f"(mean within-time median pairwise distance)")
        elif _map_type == 'accurate_kernel':
            # Defaults for the per-cell sigma estimator. Hardcoded sensible
            # values; user can override via Args.py (none of these are
            # required to be present).
            sigma_knn_k     = getattr(args, 'sigma_knn_k', 20)
            sigma_chunk     = getattr(args, 'sigma_chunk_size', 2000)
            sigma_clip_low  = getattr(args, 'sigma_clip_low', 0.05)
            sigma_clip_high = getattr(args, 'sigma_clip_high', 0.95)

            raw_sigmas = compute_per_cell_sigma_per_time(
                primary_data,
                k=sigma_knn_k,
                chunk_size=sigma_chunk,
                device=device,
            )
            clipped, sigma_lo, sigma_hi = clip_sigma_globally(
                raw_sigmas, low_q=sigma_clip_low, high_q=sigma_clip_high,
            )
            args.sigma_per_cell_primary    = clipped
            args.sigma_clip_bounds_primary = (sigma_lo, sigma_hi)

            # Diagnostic: report distribution + how many cells were clipped
            # at each tail (helps detect heavy-tailed or pathological cases).
            pool = torch.cat([s.flatten() for s in clipped])
            raw_pool = torch.cat([s.flatten() for s in raw_sigmas])
            n_clip_lo = (raw_pool < sigma_lo).sum().item()
            n_clip_hi = (raw_pool > sigma_hi).sum().item()
            n_total   = raw_pool.numel()
            print(f"[Training] accurate_kernel sigma stats "
                  f"(after global p{int(sigma_clip_low*100)}-"
                  f"p{int(sigma_clip_high*100)} clip):")
            print(f"           min={pool.min().item():.4f}  "
                  f"p{int(sigma_clip_low*100)}={sigma_lo:.4f}  "
                  f"median={pool.median().item():.4f}  "
                  f"p{int(sigma_clip_high*100)}={sigma_hi:.4f}  "
                  f"max={pool.max().item():.4f}")
            print(f"           clipped {n_clip_lo}/{n_total} "
                  f"({100.0*n_clip_lo/max(n_total,1):.1f}%) at lower bound, "
                  f"{n_clip_hi}/{n_total} "
                  f"({100.0*n_clip_hi/max(n_total,1):.1f}%) at upper bound")
            print(f"           per-time N: {[s.numel() for s in clipped]}")
            del raw_sigmas, raw_pool, pool

    # Pre-instantiate one SamplesLoss per timepoint (primary, on device).
    # Lifecycle: created once at training start, reused throughout. Total
    # = N_time instances per modality (lightweight; geomloss only stores config).
    sinkhorn_primal_list = [
        SamplesLoss(loss="sinkhorn", p=2, blur=b).to(device)
        for b in args.blur_per_time_primary
    ]

    # ---- Noise floor (lower bound for trainable Sinkhorn loss) -----------
    # The floor at timepoint t is Sinkhorn(batch1, batch2) where both
    # batches are i.i.d. samples from the same empirical distribution at t.
    # The estimator returns mean + std over repeated draws.
    #
    # Dual-update target uses a relaxed lower bound:
    #     target_t = mean_t + floor_buffer_std * std_t
    # This buffer accounts for per-batch variance: forcing training Sinkhorn
    # to match the expected mean exactly is overly strict because both the
    # floor estimate and the per-iter training Sinkhorn fluctuate within
    # one std of the mean. floor_buffer_std defaults to 1.0 (~68% confidence
    # band under a Gaussian assumption); set to 0.0 to recover the strict
    # mean-as-target behavior.
    floor_stats_pri = compute_noise_floor_per_time(
        primary_data, args.blur_per_time_primary,
        n_subsample=args.floor_n_sample,
        device=device, seed=args.seed,
    )
    args.floor_pri_mean = [s['mean'] for s in floor_stats_pri]
    args.floor_pri_std  = [s['std']  for s in floor_stats_pri]
    floor_buffer_std = getattr(args, 'floor_buffer_std', 0.0)
    # Per-modality multiplier on the final floor target. Use this to relax
    # the constraint when the noise floor is structurally unreachable
    # through the cross-modal map T (e.g., the diagnostic in
    # ParamVisual/check_T_bottleneck.py reports a T-OOS of 3-4x floor on
    # SyncOT data, so a multiplier around 4.0 is a reasonable starting
    # point for sync_loss=True; default 1.0 keeps the strict noise-floor
    # target for backward compatibility).
    floor_pri_mult = getattr(args, 'floor_pri_multiplier', 1.0)
    args.floor_pri = [(m + floor_buffer_std * s) * floor_pri_mult
                      for m, s in zip(args.floor_pri_mean, args.floor_pri_std)]
    print(f"[Training] primary noise floor mean per time = "
          f"{[f'{m:.4e}' for m in args.floor_pri_mean]}")
    print(f"[Training] primary noise floor std  per time = "
          f"{[f'{s:.4e}' for s in args.floor_pri_std]}")
    print(f"[Training] primary floor target = "
          f"(mean + {floor_buffer_std} * std) * {floor_pri_mult} "
          f"= {[f'{f:.4e}' for f in args.floor_pri]}")
    # Sum over t >= 1 (the initial timepoint t=0 has no marginal-matching
    # term in the loss, so its floor does not enter the dual-target sum).
    args.floor_total_pri = sum(args.floor_pri[1:])
    print(f"[Training] primary floor_total target (sum over t>=1) = "
          f"{args.floor_total_pri:.4f}")

    # Secondary: only when sync_loss=True (no need to compute or instantiate
    # otherwise).
    sinkhorn_secondary_list = None
    if args.sync_loss:
        medians_secondary = compute_per_time_median_distance(
            secondary_data, n_sample=args.median_n_sample
        )
        args.median_dist_secondary = medians_secondary
        args.blur_per_time_secondary = [max(blur_coeff * m, 1e-6) for m in medians_secondary]
        print(f"[Training] secondary median per time = "
              f"{[f'{m:.4f}' for m in medians_secondary]}")
        print(f"[Training] secondary blur   per time = "
              f"{[f'{b:.4e}' for b in args.blur_per_time_secondary]}")
        sinkhorn_secondary_list = [
            SamplesLoss(loss="sinkhorn", p=2, blur=b).to(device)
            for b in args.blur_per_time_secondary
        ]

        # Secondary noise floor (analogous to primary, mean + buffer * std).
        floor_stats_sec = compute_noise_floor_per_time(
            secondary_data, args.blur_per_time_secondary,
            n_subsample=args.floor_n_sample,
            device=device, seed=args.seed,
        )
        args.floor_sec_mean = [s['mean'] for s in floor_stats_sec]
        args.floor_sec_std  = [s['std']  for s in floor_stats_sec]
        # Secondary multiplier on the floor target (independent of primary).
        # In sync mode the T-bottleneck typically forces a value > 1.0,
        # because the kernel-projected secondary Sinkhorn cannot reach the
        # noise floor; see ParamVisual/check_T_bottleneck.py for an
        # empirical estimate of how large the multiplier needs to be.
        floor_sec_mult = getattr(args, 'floor_sec_multiplier', 1.0)
        args.floor_sec = [(m + floor_buffer_std * s) * floor_sec_mult
                          for m, s in zip(args.floor_sec_mean, args.floor_sec_std)]
        print(f"[Training] secondary noise floor mean per time = "
              f"{[f'{m:.4e}' for m in args.floor_sec_mean]}")
        print(f"[Training] secondary noise floor std  per time = "
              f"{[f'{s:.4e}' for s in args.floor_sec_std]}")
        print(f"[Training] secondary floor target = "
              f"(mean + {floor_buffer_std} * std) * {floor_sec_mult} "
              f"= {[f'{f:.4e}' for f in args.floor_sec]}")
        args.floor_total_sec = sum(args.floor_sec[1:])
        print(f"[Training] secondary floor_total target (sum over t>=1) = "
              f"{args.floor_total_sec:.4f}")

        del medians_secondary

    del medians_primary

    # ---------- Adaptive lambda (Lagrangian dual ascent) initialization ----
    # Only activate when args.adaptive_lambda is explicitly True. When the
    # flag is missing or False, training uses the resolved fixed
    # args.pdf_coefficient_{pri,sec} as the Sinkhorn weight per modality
    # (backward compatible: fall back to the shared args.pdf_coefficient).
    #
    # Per-modality split (Option A from notes): each fixed-mode coefficient
    # and each adaptive-mode hyperparameter (lambda_init, lambda_min,
    # lambda_max, eta_lambda) can be independently overridden for primary
    # vs secondary via the _pri / _sec suffix. When the suffixed version
    # is not provided, it falls back to the shared scalar. This lets users
    # account for different Sinkhorn scales / floor levels in the two
    # spaces without forcing them to set everything explicitly.
    args.pdf_coefficient_pri = getattr(
        args, 'pdf_coefficient_pri', float(args.pdf_coefficient))
    args.pdf_coefficient_sec = getattr(
        args, 'pdf_coefficient_sec', float(args.pdf_coefficient))
    print(f"[Training] pdf_coefficient resolved per modality: "
          f"pri={args.pdf_coefficient_pri}, sec={args.pdf_coefficient_sec}")

    if getattr(args, 'adaptive_lambda', False):
        # Shared defaults for dual-ascent hyperparameters. These are used
        # as fall-back values when the user has not provided a
        # modality-specific version. They are also kept on args so older
        # code paths that read the shared name still find a value.
        if not hasattr(args, 'lambda_init'):
            args.lambda_init = float(args.pdf_coefficient)
            print(f"[Training] args.lambda_init not set, default = "
                  f"args.pdf_coefficient = {args.lambda_init}")
        if not hasattr(args, 'lambda_min'):
            args.lambda_min = 5.0
            print(f"[Training] args.lambda_min not set, default = {args.lambda_min}")
        if not hasattr(args, 'lambda_max'):
            args.lambda_max = 100.0
            print(f"[Training] args.lambda_max not set, default = {args.lambda_max}")
        if not hasattr(args, 'eta_lambda'):
            args.eta_lambda = 10.0
            print(f"[Training] args.eta_lambda not set, default = {args.eta_lambda}")
        if not hasattr(args, 'dual_warmup'):
            args.dual_warmup = 2000
            print(f"[Training] args.dual_warmup not set, default = {args.dual_warmup}")
        if not hasattr(args, 'dual_decay'):
            args.dual_decay = 1.0
            print(f"[Training] args.dual_decay not set, default = {args.dual_decay}")

        # Per-modality init. Falls back to shared lambda_init when the
        # suffixed override is not set, which in turn defaults to the
        # per-modality pdf_coefficient resolved above.
        args.lambda_init_pri = getattr(
            args, 'lambda_init_pri',
            getattr(args, 'lambda_init', float(args.pdf_coefficient_pri)))
        args.lambda_init_sec = getattr(
            args, 'lambda_init_sec',
            getattr(args, 'lambda_init', float(args.pdf_coefficient_sec)))

        # Per-modality bounds. Fall back to shared bounds when not set.
        args.lambda_min_pri = getattr(args, 'lambda_min_pri', args.lambda_min)
        args.lambda_max_pri = getattr(args, 'lambda_max_pri', args.lambda_max)
        args.lambda_min_sec = getattr(args, 'lambda_min_sec', args.lambda_min)
        args.lambda_max_sec = getattr(args, 'lambda_max_sec', args.lambda_max)

        # Per-modality dual step.
        args.eta_lambda_pri = getattr(args, 'eta_lambda_pri', args.eta_lambda)
        args.eta_lambda_sec = getattr(args, 'eta_lambda_sec', args.eta_lambda)

        # Initialize scalar dual variables. Clip to the per-modality bounds
        # so the initial value respects each cap; without this clip, an
        # over-large init would silently exceed lambda_max during the
        # warmup phase (before the first dual update clip kicks in).
        init_pri = float(args.lambda_init_pri)
        clipped_pri = max(args.lambda_min_pri,
                          min(args.lambda_max_pri, init_pri))
        if clipped_pri != init_pri:
            print(f"[Training] WARNING: lambda_init_pri={init_pri} outside "
                  f"[{args.lambda_min_pri}, {args.lambda_max_pri}], "
                  f"clipped to {clipped_pri}")
        args.lambda_pri = clipped_pri

        if args.sync_loss:
            init_sec = float(args.lambda_init_sec)
            clipped_sec = max(args.lambda_min_sec,
                              min(args.lambda_max_sec, init_sec))
            if clipped_sec != init_sec:
                print(f"[Training] WARNING: lambda_init_sec={init_sec} outside "
                      f"[{args.lambda_min_sec}, {args.lambda_max_sec}], "
                      f"clipped to {clipped_sec}")
            args.lambda_sec = clipped_sec
        else:
            args.lambda_sec = None

        print(f"[Training] adaptive_lambda=True. "
              f"λ_pri init = {args.lambda_pri:.3f} "
              f"(bounds [{args.lambda_min_pri}, {args.lambda_max_pri}], "
              f"η={args.eta_lambda_pri})")
        if args.sync_loss:
            print(f"           λ_sec init = {args.lambda_sec:.3f} "
                  f"(bounds [{args.lambda_min_sec}, {args.lambda_max_sec}], "
                  f"η={args.eta_lambda_sec})")
    else:
        print(f"[Training] adaptive_lambda is False (or not set). "
              f"Using fixed pdf_coefficient_pri={args.pdf_coefficient_pri}, "
              f"pdf_coefficient_sec={args.pdf_coefficient_sec}.")

    # ---------- Time grid for ODE integration ----------
    time_steps_list = _build_time_steps_list(args.time_points, args.dt, device)

    # ---------- Neural ODE velocity field ----------
    func = MLPVectorField(
        dim=args.otdim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        activation=args.activation,
        unbalanced=args.unbalancedModel,
        alpha_growth=args.alpha_growth,
    ).to(device)

    # ---------- Optimizer & scheduler ----------
    optimizer = optim.AdamW(func.parameters(), lr=args.lr, weight_decay=0.001)

    total_steps = args.niters
    gamma = (1e-5 / args.lr) ** (1.0 / total_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    loss_meter = RunningAverageMeter()

    # ---------- Optional warm-start ----------
    # Load model weights only; keep optimizer state fresh so Adam's moments
    # don't carry over from a different loss configuration.
    if warm_start_path is not None:
        print(f"[Training] warm-starting from checkpoint: {warm_start_path}")
        load_checkpoint(func, args, ckpt_path=warm_start_path)

    # ---------- Training ----------
    results = train_path_epoch(
        func=func,
        primary_data=primary_data,
        time_points=args.time_points,
        time_steps_list=time_steps_list,
        optimizer=optimizer,
        device=device,
        secondary_data=secondary_data,
        MapModel=map_model,
        metric_primal=metric_primal,
        metric_secondary=metric_secondary,
        args=args,
        loss_meter=loss_meter,
        scheduler=scheduler,
        sinkhorn_primal_list=sinkhorn_primal_list,
        sinkhorn_secondary_list=sinkhorn_secondary_list,
    )

    return {
        'func': func,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'results': results,
    }
