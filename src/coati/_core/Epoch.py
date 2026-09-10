from pickletools import optimize
from syslog import LOG_ERR
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import gc
from .DataLoad import create_mixture_gaussian
from .TrainLoss import short_term_loss
import shutil
from .utility import log_losses_to_tensorboard, build_exp_name



def train_path_epoch(func, primary_data, time_points, time_steps_list,
               optimizer, device, args, MapModel = None, metric_primal = None, metric_secondary = None, secondary_data=None, writer=None, loss_meter=None, scheduler=None, rare_files=None,
               sinkhorn_primal_list=None, sinkhorn_secondary_list=None):
    
    try:
        func.train()
        epoch = 0
        
        # ============================================================
        # TensorBoard log directory.
        # Each experiment writes to a per-sweep subdirectory named by
        # build_exp_name(args), which shares the same scheme as checkpoint
        # filenames. This way sweeps over hyperparameters don't overwrite
        # each other's logs, and `tensorboard --logdir <results_dir>/tensorboard`
        # lists every experiment as a separate run.
        # Only this experiment's own subdir is wiped on rerun to avoid
        # mixing event files; the shared `args.results_dir` stays intact.
        # ============================================================
        # Ensure the top-level results directory exists (create if missing).
        os.makedirs(args.results_dir, exist_ok=True)

        exp_name = build_exp_name(args)
        tb_dir = os.path.join(args.results_dir, 'tensorboard', exp_name)
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        
        progress_bar = tqdm(range(1, args.niters + 1), desc=f"Epoch {epoch}")

        # ====================================================================
        # EMA tracking for Sinkhorn losses (Phase 2 passive monitoring).
        #
        # Track an exponential moving average (EMA) of the total Sinkhorn loss
        # (sum across segments) per modality, and compare against the noise
        # floor target stored on args (args.floor_total_pri / floor_total_sec).
        # The EMA reduces single-batch noise so the convergence trend toward
        # the floor is visible. Phase 2 only prints; Phase 3 will use these
        # EMAs to drive dual-ascent updates of lambda.
        # ====================================================================
        sink_total_pri_ema = None
        sink_total_sec_ema = None
        ema_alpha    = getattr(args, 'dual_ema_alpha', 0.1)
        monitor_freq = getattr(args, 'dual_update_freq', 10)
        tau_pri      = getattr(args, 'floor_total_pri', None)
        tau_sec      = getattr(args, 'floor_total_sec', None)

        # Per-segment EMA tracking for finer-grained monitoring.
        # ema_pri_seg[t] tracks the EMA of the Sinkhorn loss for segment t
        # (segment t covers ρ_t -> ρ_{t+1}, with marginal matching at the
        # terminal time t+1). Updated inside the t_idx loop. Used for
        # printing per-segment gap and writing per-segment tensorboard
        # scalars; does NOT participate in the dual update yet.
        N_seg = len(time_points) - 1
        ema_pri_seg = [None] * N_seg
        ema_sec_seg = [None] * N_seg if args.sync_loss else None

        for itr in progress_bar:

            
            Whole_results =  {
                'forward_density_loss': 0.0,
                'forward_mmd_loss': 0.0,
                'forward_sinkhorn_pdf_loss': 0.0,
                'forward_energy_loss': 0.0,
                'forward_energy_velocity': 0.0,
                'forward_energy_growth': 0.0,
                'forward_mass_loss': 0.0,
                'sync_secondary_energy_loss': 0.0,
                'sync_mmd_loss': 0.0,
                'sync_pdf_loss': 0.0,
                'sync_density_loss': 0.0,
                'reverse_matching_loss': 0.0,
            }
            # ============================================================
            # Resolve the Sinkhorn coefficient for this iteration.
            # When adaptive_lambda is on, use the dual variables on args
            # (updated every dual_update_freq iters by the dual-ascent step
            # at the end of the previous iter). Otherwise fall back to the
            # per-modality fixed coefficients args.pdf_coefficient_{pri,sec}
            # (resolved in Training.py with backward-compatible fallback to
            # the shared args.pdf_coefficient).
            # ============================================================
            if getattr(args, 'adaptive_lambda', False):
                pdf_coef_pri = args.lambda_pri
                pdf_coef_sec = args.lambda_sec if args.sync_loss else None
            else:
                pdf_coef_pri = args.pdf_coefficient_pri
                pdf_coef_sec = (args.pdf_coefficient_sec
                                if args.sync_loss else None)

            # short term loss
            previous_sample = None
            for t_idx in range(len(time_points) - 1):
                optimizer.zero_grad()
                if t_idx == 0:
                    previous_sample = None
                short_results, previous_sample = short_term_loss(
                    func=func,
                    t_idx=t_idx,
                    time_points=time_points,
                    primary_data=primary_data,
                    secondary_data=secondary_data,
                    time_steps_list=time_steps_list,
                    device=device,
                    MapModel=MapModel,
                    metric_primal=metric_primal,
                    metric_secondary=metric_secondary,
                    args=args,
                    itr=itr + epoch * args.niters,
                    rare_files=rare_files,
                    previous_sample=previous_sample,
                    sinkhorn_primal_list=sinkhorn_primal_list,
                    sinkhorn_secondary_list=sinkhorn_secondary_list,
                )
                
                for k in short_results:
                        Whole_results[k] += short_results[k]

                # Per-segment EMA update for monitoring (no effect on loss).
                _sp_t = short_results['forward_sinkhorn_pdf_loss']
                _sp_curr = (_sp_t.item() if torch.is_tensor(_sp_t)
                            else float(_sp_t))
                if ema_pri_seg[t_idx] is None:
                    ema_pri_seg[t_idx] = _sp_curr
                else:
                    ema_pri_seg[t_idx] = ((1.0 - ema_alpha) * ema_pri_seg[t_idx]
                                          + ema_alpha * _sp_curr)
                if args.sync_loss and ema_sec_seg is not None:
                    _ss_t = short_results['sync_pdf_loss']
                    _ss_curr = (_ss_t.item() if torch.is_tensor(_ss_t)
                                else float(_ss_t))
                    if ema_sec_seg[t_idx] is None:
                        ema_sec_seg[t_idx] = _ss_curr
                    else:
                        ema_sec_seg[t_idx] = ((1.0 - ema_alpha) * ema_sec_seg[t_idx]
                                              + ema_alpha * _ss_curr)

                primal_terminal_loss = short_results['forward_sinkhorn_pdf_loss'] * pdf_coef_pri
                
                if args.unbalancedModel:
                    primal_opt_loss = (
                        short_results['forward_density_loss'] * args.density_coefficient 
                        + short_results['forward_energy_velocity'] * args.energy_coefficient 
                    )
                else:
                    primal_opt_loss = (
                        short_results['forward_density_loss'] * args.density_coefficient 
                        + short_results['forward_energy_loss'] * args.energy_coefficient 
                    )


                short_term_total_loss = 0
                if args.unbalancedModel and args.mass_loss:
                    short_term_total_loss = short_term_total_loss + short_results['forward_mass_loss'] * args.mass_coefficient + short_results['forward_energy_growth'] * args.energy_coefficient # growth_primal = growth_secondary

                if args.sync_loss:
                    sync_loss_opt = (
                        short_results['sync_secondary_energy_loss'] * args.energy_coefficient 
                        + short_results['sync_density_loss'] * args.density_coefficient 
                    ) * args.sync_weight
                    primal_loss_opt = primal_opt_loss * (1-args.sync_weight)
                    short_term_total_loss = short_term_total_loss + sync_loss_opt + primal_loss_opt
                    
                    if args.sync_pdf_loss:
                        sec_terminal_loss = short_results['sync_pdf_loss'] * pdf_coef_sec #* args.sync_weight
                        main_terminal_loss = primal_terminal_loss #* (1-args.sync_weight)
                        short_term_total_loss = short_term_total_loss + sec_terminal_loss + main_terminal_loss
                    else:
                        short_term_total_loss = short_term_total_loss + primal_terminal_loss #* (1-args.sync_weight)

                else:
                    short_term_total_loss = short_term_total_loss + primal_opt_loss + primal_terminal_loss
            

                
                # ################################################################
                # ##  DEBUG: GRADIENT ANALYSIS — REMOVE AFTER DEBUGGING        ##
                # ################################################################
                if t_idx == 0 and itr % 100 == 0:

                    def _grad_norm():
                        return sum(
                            p.grad.norm().item() ** 2
                            for p in func.parameters() if p.grad is not None
                        ) ** 0.5

                    def _grad_of(loss_tensor):
                        optimizer.zero_grad()
                        # Disabled losses are scalar zeros; diagnostics have no gradient.
                        if not torch.is_tensor(loss_tensor) or not loss_tensor.requires_grad:
                            return 0.0
                        loss_tensor.backward(retain_graph=True)
                        return _grad_norm()

                    # ---- Primary space (always computed) ----
                    if args.unbalancedModel:
                        pri_energy_term = short_results['forward_energy_velocity'] * args.energy_coefficient
                    else:
                        pri_energy_term = short_results['forward_energy_loss'] * args.energy_coefficient
                    pri_density_term = short_results['forward_density_loss'] * args.density_coefficient
                    # Use the same Sinkhorn coefficient as the actual loss
                    # computation for this iteration (pdf_coef_pri = either
                    # args.lambda_pri when adaptive_lambda is on, or
                    # args.pdf_coefficient otherwise). Mirroring the loss
                    # makes the printed grad ratios reflect what the
                    # optimizer actually sees.
                    pri_pdf_term     = short_results['forward_sinkhorn_pdf_loss'] * pdf_coef_pri

                    g_pri_energy  = _grad_of(pri_energy_term)
                    g_pri_density = _grad_of(pri_density_term)
                    g_pri_pdf     = _grad_of(pri_pdf_term)

                    # ---- Secondary space (only if sync) ----
                    g_sec_energy = g_sec_density = g_sec_pdf = None
                    if args.sync_loss:
                        sec_energy_term  = short_results['sync_secondary_energy_loss'] * args.energy_coefficient
                        sec_density_term = short_results['sync_density_loss'] * args.density_coefficient
                        g_sec_energy  = _grad_of(sec_energy_term)
                        g_sec_density = _grad_of(sec_density_term)
                        if args.sync_pdf_loss:
                            # Same: use pdf_coef_sec to match actual loss.
                            sec_pdf_term = short_results['sync_pdf_loss'] * pdf_coef_sec
                            g_sec_pdf = _grad_of(sec_pdf_term)

                    # ---- Print ----
                    bar = "=" * 78
                    mode = "SYNC" if args.sync_loss else "NO-SYNC"
                    print(f"\n{bar}")
                    print(f"  [GRAD DEBUG | {mode}] itr={itr}, t_idx={t_idx}")
                    print(f"{bar}")
                    print(f"  {'TERM':<22}{'GRAD NORM':>14}{'/ pri_energy':>18}")
                    print(f"  {'-'*22}{'-'*14:>14}{'-'*18:>18}")
                    print(f"  {'PRIMARY':<22}")
                    print(f"    {'energy':<20}{g_pri_energy:>14.6f}{1.0:>18.4f}")
                    print(f"    {'density':<20}{g_pri_density:>14.6f}{g_pri_density/(g_pri_energy+1e-12):>18.4f}")
                    print(f"    {'pdf':<20}{g_pri_pdf:>14.6f}{g_pri_pdf/(g_pri_energy+1e-12):>18.4f}")
                    if args.sync_loss:
                        print(f"  {'SECONDARY':<22}")
                        print(f"    {'energy':<20}{g_sec_energy:>14.6f}{g_sec_energy/(g_pri_energy+1e-12):>18.4f}")
                        print(f"    {'density':<20}{g_sec_density:>14.6f}{g_sec_density/(g_pri_energy+1e-12):>18.4f}")
                        if g_sec_pdf is not None:
                            print(f"    {'pdf':<20}{g_sec_pdf:>14.6f}{g_sec_pdf/(g_pri_energy+1e-12):>18.4f}")
                        print(f"  {'-'*54}")
                        print(f"    pri_energy / sec_energy = {g_pri_energy/(g_sec_energy+1e-12):>.4f}")
                    print(f"{bar}\n")

                    optimizer.zero_grad()  # clean up before the real backward
                # ################################################################
                # ##  END DEBUG                                                ##
                # ################################################################

                short_term_total_loss.backward()
                optimizer.step()

            # ============================================================
            # EMA update for total Sinkhorn loss across segments
            # (Phase 2 passive monitoring; no effect on training behavior).
            # ============================================================
            sp_t = Whole_results['forward_sinkhorn_pdf_loss']
            sp_curr = sp_t.item() if torch.is_tensor(sp_t) else float(sp_t)
            if sink_total_pri_ema is None:
                sink_total_pri_ema = sp_curr
            else:
                sink_total_pri_ema = ((1.0 - ema_alpha) * sink_total_pri_ema
                                      + ema_alpha * sp_curr)

            if args.sync_loss:
                ss_t = Whole_results['sync_pdf_loss']
                ss_curr = ss_t.item() if torch.is_tensor(ss_t) else float(ss_t)
                if sink_total_sec_ema is None:
                    sink_total_sec_ema = ss_curr
                else:
                    sink_total_sec_ema = ((1.0 - ema_alpha) * sink_total_sec_ema
                                          + ema_alpha * ss_curr)

            # Periodic print: show EMA-smoothed Sinkhorn vs floor target.
            if itr % monitor_freq == 0 and tau_pri is not None:
                gap_pri = sink_total_pri_ema - tau_pri
                print(f"\n[EMA monitor] iter={itr}")
                print(f"  primary:   S_total_ema={sink_total_pri_ema:.4f}, "
                      f"tau={tau_pri:.4f}, gap={gap_pri:+.4f}")
                if args.sync_loss and tau_sec is not None:
                    gap_sec = sink_total_sec_ema - tau_sec
                    print(f"  secondary: S_total_ema={sink_total_sec_ema:.4f}, "
                          f"tau={tau_sec:.4f}, gap={gap_sec:+.4f}")

                # Per-segment breakdown (monitoring only, no dual update).
                if args.floor_pri is not None:
                    print(f"  per-segment primary:")
                    for t in range(N_seg):
                        if ema_pri_seg[t] is None:
                            continue
                        floor_t = args.floor_pri[t + 1]
                        gap_t = ema_pri_seg[t] - floor_t
                        print(f"    seg{t} ({t}->{t+1}): "
                              f"S_ema={ema_pri_seg[t]:.4e}, "
                              f"tau={floor_t:.4e}, gap={gap_t:+.4e}")
                if (args.sync_loss and ema_sec_seg is not None
                    and getattr(args, 'floor_sec', None) is not None):
                    print(f"  per-segment secondary:")
                    for t in range(N_seg):
                        if ema_sec_seg[t] is None:
                            continue
                        floor_t = args.floor_sec[t + 1]
                        gap_t = ema_sec_seg[t] - floor_t
                        print(f"    seg{t} ({t}->{t+1}): "
                              f"S_ema={ema_sec_seg[t]:.4e}, "
                              f"tau={floor_t:.4e}, gap={gap_t:+.4e}")

            # ============================================================
            # Dual ascent update for lambda (Phase 3).
            # Active only when args.adaptive_lambda=True. Updates after
            # warmup, every dual_update_freq iterations, using the
            # EMA-smoothed total Sinkhorn vs the precomputed floor target.
            #
            #     violation = S_total_ema - tau_total
            #     λ_new = max(λ_min, min(λ_max, λ * decay + η * violation))
            #
            # When violation > 0 (Sinkhorn above floor, marginal not yet
            # matched), λ grows to push harder. Once violation ≈ 0, λ
            # stabilizes. lambda_max acts as a safety cap to prevent
            # runaway when the floor is structurally unreachable.
            # ============================================================
            if (getattr(args, 'adaptive_lambda', False)
                and itr > getattr(args, 'dual_warmup', 2000)
                and itr % monitor_freq == 0
                and tau_pri is not None):

                # Per-modality bounds and step (set in Training.py via
                # _pri / _sec suffix; fall back here to the shared scalar
                # in case Epoch.py is invoked without the upstream
                # Training.py resolve step having run).
                lam_min_pri = getattr(args, 'lambda_min_pri',
                                      getattr(args, 'lambda_min', 5.0))
                lam_max_pri = getattr(args, 'lambda_max_pri',
                                      getattr(args, 'lambda_max', 100.0))
                eta_lam_pri = getattr(args, 'eta_lambda_pri',
                                      getattr(args, 'eta_lambda', 100.0))
                lam_min_sec = getattr(args, 'lambda_min_sec',
                                      getattr(args, 'lambda_min', 5.0))
                lam_max_sec = getattr(args, 'lambda_max_sec',
                                      getattr(args, 'lambda_max', 100.0))
                eta_lam_sec = getattr(args, 'eta_lambda_sec',
                                      getattr(args, 'eta_lambda', 100.0))
                eta_neg_scale = getattr(args, 'eta_neg_scale', 100.0)
                decay   = getattr(args, 'dual_decay', 1.0)
                # 'max' (default): drive lambda by the worst-segment
                #   violation max_t (S_t - tau_t). Single lambda is shared
                #   by all segments in the loss, but the update strategy
                #   enforces per-segment constraint S_t <= tau_t for every
                #   t (L_inf relaxation). Lambda only stops growing when
                #   every segment is at/below its own floor.
                # 'sum': legacy aggregate behavior. Drive lambda by total
                #   sinkhorn vs total floor. Allows easy segments to
                #   compensate for hard ones.
                update_mode = getattr(args, 'lambda_update_mode', 'max')

                # Primary dual update.
                # Asymmetric step size: when violation < 0 (overshoot, sink
                # below floor), use a larger effective eta so that lambda
                # shrinks faster. This counteracts the natural asymmetry
                # where positive violations can be large (sinkhorn far above
                # floor) but negative violations are bounded by -tau.
                worst_seg_pri = None
                if update_mode == 'max':
                    violations_pri_seg = [
                        ema_pri_seg[t] - args.floor_pri[t + 1]
                        for t in range(N_seg)
                    ]
                    worst_seg_pri = max(range(N_seg),
                                        key=lambda t: violations_pri_seg[t])
                    violation_pri = violations_pri_seg[worst_seg_pri]
                else:
                    violation_pri = sink_total_pri_ema - tau_pri
                eta_eff_pri = (eta_lam_pri if violation_pri > 0
                               else eta_lam_pri * eta_neg_scale)
                new_lam_pri = args.lambda_pri * decay + eta_eff_pri * violation_pri
                args.lambda_pri = float(max(lam_min_pri,
                                            min(lam_max_pri, new_lam_pri)))

                # Secondary dual update (same asymmetric step, independent
                # bounds and step from the secondary-specific args).
                worst_seg_sec = None
                if args.sync_loss and tau_sec is not None:
                    if update_mode == 'max' and ema_sec_seg is not None:
                        violations_sec_seg = [
                            ema_sec_seg[t] - args.floor_sec[t + 1]
                            for t in range(N_seg)
                        ]
                        worst_seg_sec = max(range(N_seg),
                                            key=lambda t: violations_sec_seg[t])
                        violation_sec = violations_sec_seg[worst_seg_sec]
                    else:
                        violation_sec = sink_total_sec_ema - tau_sec
                    eta_eff_sec = (eta_lam_sec if violation_sec > 0
                                   else eta_lam_sec * eta_neg_scale)
                    new_lam_sec = args.lambda_sec * decay + eta_eff_sec * violation_sec
                    args.lambda_sec = float(max(lam_min_sec,
                                                min(lam_max_sec, new_lam_sec)))

                # Log the update.
                pri_extra = (f", worst_seg={worst_seg_pri}"
                             if worst_seg_pri is not None else "")
                msg = (f"[Dual:{update_mode}] iter={itr}: "
                       f"λ_pri={args.lambda_pri:.3f} "
                       f"(violation={violation_pri:+.4f}{pri_extra})")
                if args.sync_loss and tau_sec is not None:
                    sec_extra = (f", worst_seg={worst_seg_sec}"
                                 if worst_seg_sec is not None else "")
                    msg += (f", λ_sec={args.lambda_sec:.3f} "
                            f"(violation={violation_sec:+.4f}{sec_extra})")
                print(msg)

                # Saturation warnings: if λ pinned at lambda_max, the floor
                # target is likely unreachable; raising lambda_max further
                # will not help, the user should relax the target instead.
                if args.lambda_pri >= lam_max_pri - 1e-3:
                    print(f"  WARNING: primary λ saturated at lambda_max_pri="
                          f"{lam_max_pri}. Floor likely unreachable.")
                if (args.sync_loss and tau_sec is not None
                    and args.lambda_sec >= lam_max_sec - 1e-3):
                    print(f"  WARNING: secondary λ saturated at lambda_max_sec="
                          f"{lam_max_sec}. Floor likely unreachable.")

            # ============================================================
            # Tensorboard logging for adaptive-lambda monitoring.
            # We log EMA + tau + gap regardless of adaptive_lambda (always
            # useful as diagnostics), and λ only when adaptive_lambda=True.
            # ============================================================
            if writer is not None:
                if sink_total_pri_ema is not None and tau_pri is not None:
                    writer.add_scalar('Sinkhorn/primary_S_total_ema',
                                      sink_total_pri_ema, itr)
                    writer.add_scalar('Sinkhorn/primary_tau', tau_pri, itr)
                    writer.add_scalar('Sinkhorn/primary_gap',
                                      sink_total_pri_ema - tau_pri, itr)
                if (args.sync_loss and sink_total_sec_ema is not None
                    and tau_sec is not None):
                    writer.add_scalar('Sinkhorn/secondary_S_total_ema',
                                      sink_total_sec_ema, itr)
                    writer.add_scalar('Sinkhorn/secondary_tau', tau_sec, itr)
                    writer.add_scalar('Sinkhorn/secondary_gap',
                                      sink_total_sec_ema - tau_sec, itr)

                # Per-segment gap only: tensorboard panels would explode if
                # we logged ema + tau + gap for every segment. Gap alone is
                # the diagnostic that matters (tau is constant, ema is
                # implied by gap). Useful for spotting which segment is
                # failing to reach its floor.
                if getattr(args, 'floor_pri', None) is not None:
                    for t in range(N_seg):
                        if ema_pri_seg[t] is None:
                            continue
                        floor_t = args.floor_pri[t + 1]
                        writer.add_scalar(f'SinkhornSeg/primary_seg{t}_gap',
                                          ema_pri_seg[t] - floor_t, itr)
                if (args.sync_loss and ema_sec_seg is not None
                    and getattr(args, 'floor_sec', None) is not None):
                    for t in range(N_seg):
                        if ema_sec_seg[t] is None:
                            continue
                        floor_t = args.floor_sec[t + 1]
                        writer.add_scalar(f'SinkhornSeg/secondary_seg{t}_gap',
                                          ema_sec_seg[t] - floor_t, itr)

                if getattr(args, 'adaptive_lambda', False):
                    writer.add_scalar('Lambda/lambda_pri',
                                      args.lambda_pri, itr)
                    if args.sync_loss and args.lambda_sec is not None:
                        writer.add_scalar('Lambda/lambda_sec',
                                          args.lambda_sec, itr)

            log_losses_to_tensorboard(writer, Whole_results, itr, args)

            
            if args.train_dir is not None:
                # ============================================================
                # Save checkpoint every 5000 iterations.
                # The filename is `ckpt_{exp_name}_iter{itr}.pth`, where
                # `exp_name` comes from build_exp_name(args) and encodes the
                # sweep-relevant hyperparameters (seed, energy/pdf/density
                # coefficients, and sync_weight when sync_loss is enabled).
                # Use the same exp_name for the TensorBoard log subdirectory
                # so checkpoints and TB runs line up across sweeps.
                # ============================================================
                if itr % 500 == 0:
                    if not os.path.exists(args.train_dir):
                        os.makedirs(args.train_dir, exist_ok=True)
                    exp_name = build_exp_name(args)
                    ckpt_name = f"ckpt_{exp_name}_iter{itr}.pth"
                    ckpt_path = os.path.join(args.train_dir, ckpt_name)
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'iteration': itr
                    }, ckpt_path)
                    print('Stored checkpoint at {}'.format(ckpt_path))
                    # Calculate average terminal density loss
                    
                if itr == args.niters: 
                    final_ckpt_path = os.path.join(args.train_dir, f'path_param.pth')
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'iteration': itr
                    }, final_ckpt_path)
                    print('Stored final checkpoint at {}'.format(final_ckpt_path))
                
            if itr % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Iter {itr}: Current Learning Rate: {current_lr}')
                print('----------------------------------')
            
                        
                        
        writer.close()


    except KeyboardInterrupt:

        os.makedirs(args.train_dir, exist_ok=True)
        ckpt_path = os.path.join(args.train_dir, 'ckptBreak.pth')
        torch.save({
            'func_state_dict': func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        print('Stored ckpt at {}'.format(ckpt_path))
              

    print('Training complete after {} iters.'.format(itr))


from .FlowMatching import *
from .utility import load_checkpoint

def train_CFM_epoch(func, primary_data, time_points,
               optimizer, device, args, writer=None, loss_meter=None, scheduler=None):
    
    writer = SummaryWriter(log_dir=args.results_dir)
    epoch = 0
    FM = ConditionalFlowMatcher(sigma=args.sigma)
    time_points = torch.tensor(time_points, dtype = torch.float32, device=device)
    ot_plans, sampling_info = compute_pairing_uot_plans(primary_data, time_points,use_mini_batch=True, chunk_size=1000, reg_strategy="per_time", device = device)

    progress_bar = tqdm(range(1, args.CFM_niters + 1), desc=f"Epoch {epoch}")
    load_checkpoint(func, args, ckpt_path='path_param.pth')
        
    for itr in progress_bar:
        optimizer.zero_grad()
        loss, penalty = train_flow_matching_epoch(
            func,
            FM, primary_data, time_points,
            batch_size = args.CFM_mini_batch_size,
            ot_plans = ot_plans,
            sampling_info = sampling_info,
            device = device
        )
        
        # Stop training if loss becomes NaN (numerical instability)
        if torch.isnan(loss):
            print("Training stopped due to NaN loss")
            break

        total_loss = loss
        total_loss.backward()
            # Update optimizer
        optimizer.step()
            # Update scheduler if initialized
        if scheduler is not None:
            scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()
        loss_meter.update(total_loss)
        
        if writer is not None:
            writer.add_scalar('CFM/total', loss, itr)
        
        if args.train_dir is not None:
                # Save checkpoint every 100 iterations
                if itr % 100 == 0:
                    if not os.path.exists(args.train_dir):
                        os.makedirs(args.train_dir, exist_ok=True)
                    # Save with iteration number in filename
                    ckpt_path = os.path.join(args.train_dir, f'cfm_iter_{itr}.pth')
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'iteration': itr,
                    }, ckpt_path)
                    print('Stored checkpoint at {}'.format(ckpt_path))
                if itr == args.CFM_niters:
                    final_ckpt_path = os.path.join(args.train_dir, f'cfm_last.pth')
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'iteration': itr,
                    }, final_ckpt_path)
                    print('Stored final checkpoint at {}'.format(final_ckpt_path))

    writer.close()

    return



    



def train_flow_matching_epoch(func, FM, X, time, 
                             batch_size, ot_plans, sampling_info, device):
        """Calculate loss for one epoch of Flow Matching training
        
        Args:
            FM: ConditionalRegularizedUnbalancedFlowMatcher instance
            X: List of numpy arrays where each element represents samples at a specific time point
            time: Tensor of time points (device-compatible)
            lambda_pen: Penalty weight for score network training
            batch_size: Batch size for sampling
            uot_plans: Precomputed UOT plans for sampling
            sampling_info: Additional sampling information from compute_uot_plans
            regress_v: Flag to train velocity network (v)
            regress_g: Flag to train growth network (g)
            regress_score: Flag to train score network
        
        Returns:
            tuple: (total_loss, penalty) where both are torch tensors
        """
        # Reset gradients before each batch
        
        # Sample batch data for Flow Matching (time, positions, velocities, growth values, weights, noise)
        t, xt, ut, eps = get_batch_ot_fm(FM, X, time, batch_size, ot_plans, sampling_info, device=device)
        # Reshape time tensor to (batch_size, 1) for concatenation with position data
        t = torch.unsqueeze(t, 1).to(device)

        '''
        # Compute lambda(t) (time-dependent weighting factor for score network)
        t_floor = torch.zeros_like(t)
        t_ceil = torch.zeros_like(t)
        # Determine time interval bounds (t_floor and t_ceil) for each sample in the batch
        for j in range(len(time) - 1):
            mask = (t >= time[j]) & (t < time[j + 1])
            t_floor[mask] = time[j]
            t_ceil[mask] = time[j + 1]
        # Calculate normalized time within interval and compute lambda(t)
        lambda_t = FM.compute_lambda((t - t_floor) / (t_ceil - t_floor))
        '''

        # Enable gradient computation for position data (required for score calculation via autograd)
        xt = xt.requires_grad_(True)
        xt = xt.to(device)
        # Get references to model components
        v_net = func
        #score_net = self.model.score_net
        # Concatenate position and time data for network input (shape: batch_size × (2 + 1) = batch_size × 3)


        # Initialize loss and penalty
        loss = 0.0
        penalty = 0.0
        '''
        # Train score network if enabled
        if regress_score:
            # Predict score potential (value_st) from score network
            value_st = score_net(net_input)
            # Compute score via automatic differentiation (gradient of value_st w.r.t. xt)
            st = torch.autograd.grad(
                outputs=value_st,
                inputs=xt,
                grad_outputs=torch.ones_like(value_st),
                create_graph=True  # Required for second-order gradients (if needed)
            )[0]
            # Calculate weighted MSE loss for score network
            score_loss = torch.mean(weights * ((lambda_t[:, None] * st + eps) ** 2))
            # Handle NaN loss (set to 0 to avoid training instability)
            if torch.isnan(score_loss):
                score_loss = 0.0
            loss += score_loss
            # Add penalty term to regularize score potential (prevents exploding values)
            penalty += lambda_pen * torch.max(torch.relu(value_st))
        '''
        

        # Predict velocity from velocity network
        v_predict = v_net(t, xt)
        # Add weighted MSE loss between predicted and target velocities
        loss += torch.mean((v_predict - ut) ** 2)
        

        return loss, penalty