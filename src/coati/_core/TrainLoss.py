
from unittest import result
from numpy import divide
import torch
from typing import Tuple, List, Dict
from .Loss import calculate_density_loss, calculate_pdf_loss, calc_mass_loss
from .DataLoad import Sampling
from torchdiffeq import odeint
from .Loss import MMD_loss, sample_target, calculate_pdf_loss
from .MapSpace import map_whole_trajectory2another_manifold, smooth_trajectory, kernel_project_paired
from geomloss import SamplesLoss
from torch.utils.checkpoint import checkpoint
from .Integration import CrossModalVAE
from .RiemannMetric import ConformalMetric


import time

def tic():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return time.time()

def toc(t0, name):
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    print(f"[Time] {name}: {time.time() - t0:.4f} s")


def get_interval_mass_target(args, primary_data, t_idx):
    """
    Return the target mass ratio for one observed interval.

    By default, unbalanced training uses the observed sample-count ratio
    between consecutive snapshots. If args.biological_num is provided, its
    entries are treated as external biological cell-number priors at each
    observed time point, and the interval target is biological_num[t+1] /
    biological_num[t].
    """
    biological_num = getattr(args, 'biological_num', None)
    if biological_num is None:
        return primary_data[t_idx + 1].shape[0] / primary_data[0].shape[0]

    if isinstance(biological_num, torch.Tensor):
        biological_num = biological_num.detach().cpu().tolist()
    else:
        biological_num = list(biological_num)

    expected_len = len(primary_data)
    if len(biological_num) != expected_len:
        raise ValueError(
            f"args.biological_num must have length {expected_len}, "
            f"got {len(biological_num)}."
        )

    current_num = float(biological_num[0])
    next_num = float(biological_num[t_idx + 1])
    if current_num <= 0.0 or next_num <= 0.0:
        raise ValueError("args.biological_num entries must be positive.")

    return next_num / current_num


def build_paired_basis(primary_data, secondary_data, n_per_time, device,
                       sigma_per_cell=None):
    """
    Subsample paired (primary, secondary) cells from each time point and
    concatenate them into a single basis pool for the kernel projection.
    Pairing is enforced by reusing the same indices in both modalities.
    Returns detached tensors (basis is not optimized).

    When ``sigma_per_cell`` is provided (used by map_type='accurate_kernel'),
    the same per-time indices are also applied to slice the per-cell sigma
    list so that X_basis[j], Y_basis[j], sigma_basis[j] all refer to the
    same cell j.

    Args:
        primary_data:    list of (N_t, d_x) tensors, one per timepoint.
        secondary_data:  list of (N_t, d_y) tensors, paired with primary.
        n_per_time:      max number of cells to draw per timepoint.
        device:          target device for the basis tensors.
        sigma_per_cell:  optional list of (N_t,) cpu tensors holding the
                         per-cell adaptive bandwidth (precomputed in
                         Training.py). If None, sigma_basis is None.

    Returns:
        (X_basis, Y_basis, sigma_basis):
            X_basis:     (sum n_take, d_x) detached tensor on device.
            Y_basis:     (sum n_take, d_y) detached tensor on device.
            sigma_basis: (sum n_take,)     detached tensor on device,
                         or None when sigma_per_cell is None.
    """
    X_chunks, Y_chunks = [], []
    sigma_chunks = [] if sigma_per_cell is not None else None
    for t, (X_t, Y_t) in enumerate(zip(primary_data, secondary_data)):
        n_avail = X_t.shape[0]
        n_take = min(n_per_time, n_avail)
        idx = torch.randperm(n_avail, device=X_t.device)[:n_take]
        X_chunks.append(X_t[idx])
        Y_chunks.append(Y_t[idx])
        if sigma_per_cell is not None:
            # sigma_per_cell[t] lives on cpu; index on cpu, move to device
            # alongside the basis tensors below.
            idx_cpu = idx.cpu()
            sigma_chunks.append(sigma_per_cell[t][idx_cpu])
    X_basis = torch.cat(X_chunks, dim=0).to(device).detach()
    Y_basis = torch.cat(Y_chunks, dim=0).to(device).detach()
    if sigma_chunks is not None:
        sigma_basis = torch.cat(sigma_chunks, dim=0).to(device).detach()
    else:
        sigma_basis = None
    return X_basis, Y_basis, sigma_basis


def short_term_loss(
    func,
    t_idx: int,
    time_points: List[float],
    primary_data: List[torch.Tensor],
    secondary_data: List[torch.Tensor] = None,
    time_steps_list: List[Tuple] = None,
    device: torch.device = None,
    MapModel: CrossModalVAE = None,
    metric_primal: ConformalMetric = None,
    metric_secondary: ConformalMetric = None,
    args = None,
    itr=None,
    rare_files=None,
    previous_sample=None,
    sinkhorn_primal_list = None,
    sinkhorn_secondary_list = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Compute forward and backward losses for a single time step
    
    Parameters:
    - func: Neural ODE function
    - t_idx: Current time index
    - time_points: List of time points
    - primary_data: List of primary space data
    - secondary_data: List of secondary space data
    - time_steps_list: List of time steps
    - device: Computation device
    - args: Arguments object
    - itr: Current iteration number (optional)
    
    Returns:
    - Dictionary containing various losses
    """
    # get arguments

    mmd_criterion = MMD_loss()
    # Get current and next time point data
    z_curr_primal = primary_data[t_idx]
    z_next_primal = primary_data[t_idx + 1]
    relative_mass = get_interval_mass_target(args, primary_data, t_idx)

    if previous_sample is None:
        previous_sample = {
            'primal_x': None,
            'primal_lnw': None,
            'secondary_x': None,
        }
    
    if previous_sample['primal_x'] is not None:
        z_curr_sub = previous_sample['primal_x']
        curr_sub_idx = None
    else:
        # This is the initial sampling or the local version
        z_curr_sub, curr_sub_idx = sample_target(z_curr_primal, args.num_samples, return_index=True)
    z_next_sub, next_sub_idx = sample_target(z_next_primal, args.num_samples, return_index=True)

    # Paired secondary subsampling (same indices as primary).
    if secondary_data is not None:
        z_next_sec_sub = secondary_data[t_idx + 1][next_sub_idx]
        if curr_sub_idx is not None:
            z_curr_sec_sub = secondary_data[t_idx][curr_sub_idx]
        else:
            z_curr_sec_sub = previous_sample['secondary_x']
    else:
        z_curr_sec_sub = None
        z_next_sec_sub = None

    if args.support_points:
        z_support_sub = sample_target(primary_data[len(primary_data) - 1], args.num_samples)
        manifold_concat = torch.cat((z_curr_sub, z_next_sub, z_support_sub), dim=0)
    else:
        manifold_concat = torch.cat((z_curr_sub, z_next_sub), dim=0)

    # Paired secondary_manifold_concat — only built when sync_loss is on.
    secondary_manifold_concat = None
    if args.sync_loss and secondary_data is not None:
        if args.support_points:
            n_avail = primary_data[-1].shape[0]
            sup_idx = torch.randperm(n_avail, device=primary_data[-1].device)[:args.num_samples]
            z_support_sec_sub = secondary_data[-1][sup_idx]
            secondary_manifold_concat = torch.cat(
                (z_curr_sec_sub, z_next_sec_sub, z_support_sec_sub), dim=0
            )
        else:
            secondary_manifold_concat = torch.cat((z_curr_sec_sub, z_next_sec_sub), dim=0)
    
    
    # Get time steps
    time_steps_backward, time_steps_forward = time_steps_list[t_idx]
    
    
    # Initialize results dictionary
    results =  {
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
    if args.time_plot:
        t0 = tic()
    # Sample terminal points
    if previous_sample['primal_x'] is None:
        #This is the starting timepoint
        z_curr_primal_sample = z_curr_sub
        lnw_0 = torch.log(torch.ones(args.num_samples, 1) / args.num_samples).to(device)
    else:
        if args.MixSampling:
            new_sample, new_sample_index = sample_target(z_curr_primal, args.num_samples,  return_index = True)

            cat_sample = torch.cat((previous_sample['primal_x'], new_sample), dim=0)
            z_curr_primal_sample = sample_target(cat_sample, args.num_samples)
        else:
            new_sample = None
            z_curr_primal_sample = previous_sample['primal_x']
            lnw_0 = previous_sample['primal_lnw']


    print(f"new_sample shape for t_idx {t_idx} is", z_curr_primal_sample.shape)
    if args.unbalancedModel:
        e_v_0 = torch.zeros(args.num_samples, 1).to(device)
        e_g_0 = torch.zeros(args.num_samples, 1).to(device)
        zt_primal_sample, lnw_t, e_v_t, e_g_t = odeint(
            func,
            (z_curr_primal_sample, lnw_0, e_v_0, e_g_0),
            time_steps_forward,
            method='euler',
        )

    else:
        e_0 = torch.zeros(args.num_samples, 1).to(device)
        zt_primal_sample, e_t = odeint(
            func,
            (z_curr_primal_sample, e_0),
            time_steps_forward,
            method='euler',
        )
    
    if args.unbalancedModel and args.mass_loss:
        mass_loss = calc_mass_loss(zt_primal_sample[-1], z_next_sub, lnw_t[-1], relative_mass, args.global_mass)
        results['forward_mass_loss'] = mass_loss


    if args.time_plot:
        toc(t0, "Sampling + ODE")

    
    if args.time_plot:
        t0 = tic()
    
    time_length, Sample_size, dim_primal = zt_primal_sample.shape

    density_loss = 0 
    
    zt_trajectory_flatten = zt_primal_sample.reshape(-1, zt_primal_sample.shape[-1])
    
    density_loss = calculate_density_loss(
        zt_trajectory_flatten, 
        manifold_concat, 
        k=args.density_topk, 
        hinge_value=args.density_hinge
    )
    
    
    
    
    
    density_loss = density_loss + calculate_density_loss(
        zt_primal_sample[-1], 
        z_next_sub, 
        k=args.density_topk, 
        hinge_value=args.density_hinge
    ) * args.terminal_density_weight
    results['forward_density_loss'] = density_loss

    if args.time_plot:
        toc(t0, "Density Loss")


    if args.time_plot:
        t0 = tic()
        
    # Calculate MMD loss
    mmd_loss = 0
    if args.mmd_loss:
        next_mmd = mmd_criterion(zt_primal_sample[-1], z_next_sub)
        mmd_loss = next_mmd 
        results['forward_mmd_loss'] = mmd_loss
    if args.time_plot:
        toc(t0, "MMD Loss")
    
    if args.time_plot:
        t0 = tic()
    if args.sinkhorn_pdf:
        if args.unbalancedModel:
            loss_pdf = calculate_pdf_loss(
                zt_primal_sample[-1], z_next_sub,
                sinkhorn_loss=sinkhorn_primal_list[t_idx + 1],
                lnw_source=lnw_t[-1],
                unbalancedModel=args.unbalancedModel,
            )
        else:
            loss_pdf = calculate_pdf_loss(
                zt_primal_sample[-1], z_next_sub,
                sinkhorn_loss=sinkhorn_primal_list[t_idx + 1],
            )
        print("sinkhorn pdf loss", loss_pdf)
        results['forward_sinkhorn_pdf_loss'] = loss_pdf
    if args.time_plot:
        toc(t0, "Sinkhorn PDF Loss")

    
    '''
    if args.time_plot:
        t0 = tic()
    dz_dt_list = []
    
    # backward energy loss
    for t_i_forward, zi_primal_forward in zip(
        time_steps_forward, 
        zt_primal_sample
    ):
        #zi_primal_forward.requires_grad_(True)
        e0 = torch.zeros(zi_primal_forward.shape[0], 1).to(device)
        dz_dt_i_forward, _ = func(t_i_forward, (zi_primal_forward, e0))
        # Note: If metric is provided to MLPVectorField, de_dt already contains Riemannian energy
        # No need to compute manifold_speed separately here
        dz_dt_list.append(dz_dt_i_forward)   

    
    dz_dt_all = torch.stack(dz_dt_list, dim=0)
    #dz_dt_norm_squared = (dz_dt_all ** 2).sum(dim=-1)
    dz_dt_norm_squared = (dz_dt_all ** 2)

    if args.time_plot:
        toc(t0, "Energy Loss")
    
    
    # Calculate energy components
    energy_components = (args.dt * (dz_dt_norm_squared)  )
    energy_loss = energy_components.sum(dim=0).mean() 
    '''
    
    if args.unbalancedModel:
        energy_velocity = e_v_t[-1].sum() # because de/dt already contains the weight
        energy_growth = e_g_t[-1].sum() # because de/dt already contains the weight
        energy_loss = energy_velocity + energy_growth
        results['forward_energy_velocity'] = energy_velocity
        results['forward_energy_growth'] = energy_growth
    else:
        energy_loss = e_t[-1].mean()
    results['forward_energy_loss'] = energy_loss

    if (args.reverseMatching):
        z_reverse = Sampling(args.num_samples, device, z_next_primal, rare_files=rare_files)

        zt_reverse_sample= odeint(
            func,
            (z_reverse),
            time_steps_backward,
            atol=1e-6,
            rtol=1e-3,
            method='euler',
        )
        if args.sinkhorn_pdf:
            reverse_matching_loss = calculate_pdf_loss(
                zt_reverse_sample[-1], z_curr_sub,
                sinkhorn_loss=sinkhorn_primal_list[t_idx],
            )
            results['reverse_matching_loss'] = reverse_matching_loss
        if args.mmd_loss:
            reverse_matching_loss = mmd_criterion(zt_reverse_sample[-1], z_curr_sub)
            results['reverse_matching_loss'] = reverse_matching_loss 

        
    
     # ============ SYNCHRONIZATION LOSS CALCULATION ============
     # backward evolution
    if args.sync_loss:
        if  secondary_data is None:
            raise ValueError(
                "Secondary data is required for synchronization loss calculation"
            )

        # project the forward data to the secondary data
        mapped_mainfold_list = []
        
        time_length = zt_primal_sample.shape[0]
        if args.time_plot:
            t0 = tic()
        # Dispatch the primary -> secondary projection:
        #   args.T_model set  -> use the caller-provided frozen NN.
        #                        The forward signature must be
        #                        `forward(q, t)` where t may be silently
        #                        ignored. Architecture is entirely the
        #                        caller's choice.
        #   MapModel is None  -> non-parametric kernel projection
        #                        (kernel / accurate_kernel by sigma type).
        #   otherwise         -> legacy CrossModalVAE map_rna_to_atac.
        if getattr(args, 'T_model', None) is not None:
            # Call the external T model on the ODE-evolved trajectory
            # together with the corresponding ODE time step. Time is
            # broadcast to (T_ode, B) so it lines up with zt's leading
            # shape; a model that ignores t will accept and drop it.
            T_ode = zt_primal_sample.shape[0]
            B     = zt_primal_sample.shape[1]
            t_q = (time_steps_forward.to(zt_primal_sample.dtype)
                   .to(zt_primal_sample.device)
                   .reshape(T_ode, 1).expand(T_ode, B))
            secondary_trajectory = args.T_model(zt_primal_sample, t_q)

        elif MapModel is None:
            # Differentiable Gaussian-kernel projection on a fresh paired basis.
            # When map_type='accurate_kernel' is active, args.sigma_per_cell_primary
            # holds a list of per-time per-cell bandwidths (precomputed and
            # globally percentile-clipped in Training.py). build_paired_basis
            # subsamples it with the same indices as X_basis / Y_basis so the
            # three tensors stay row-aligned. Otherwise the scalar
            # args.kernel_sigma is used (map_type='kernel' path).
            sigma_per_cell = getattr(args, 'sigma_per_cell_primary', None)
            X_basis, Y_basis, sigma_basis = build_paired_basis(
                primary_data, secondary_data,
                n_per_time=args.num_samples,
                device=device,
                sigma_per_cell=sigma_per_cell,
            )
            sigma_arg = sigma_basis if sigma_basis is not None else args.kernel_sigma
            secondary_trajectory = kernel_project_paired(
                traj_points=zt_primal_sample,
                X_basis=X_basis,
                Y_basis=Y_basis,
                sigma=sigma_arg,
            )
            del X_basis, Y_basis
            if sigma_basis is not None:
                del sigma_basis

        else:
            dim_secondary = secondary_data[t_idx].shape[1]
            primary_traj_points = zt_primal_sample.reshape(time_length*Sample_size, dim_primal).to(device)
            secondary_trajectory = MapModel.map_rna_to_atac(primary_traj_points)
            secondary_trajectory = secondary_trajectory.reshape(time_length, Sample_size, dim_secondary)
            #smoothed_secondary_trajectory = smooth_trajectory(secondary_trajectory, window_size=7)

        if args.time_plot:
            toc(t0, "Mapping from primary to secondary")
        # secondary energy loss
        secondary_energy_loss = 0


        if args.time_plot:
            t0 = tic()
        # generate point cloud pairs
        

        if metric_secondary is None:
            diff = secondary_trajectory[1:] - secondary_trajectory[:-1]   # (T-1, N, D)
            sq = diff.pow(2).sum(dim=2)           # (T-1, N)
            secondary_energy_loss = (0.5 * sq.sum(dim=0) / args.dt).mean()
            #batch_losses = [w2_loss((pc1), (pc2)) for pc1, pc2 in point_cloud_pairs]
            #secondary_energy_loss = torch.sum((torch.stack(disp)**2), dim=0).mean()/args.dt
            results['sync_secondary_energy_loss'] = secondary_energy_loss
        else:
            T, B, D = secondary_trajectory.shape
            trajectory_1 = secondary_trajectory[:-1]
            trajectory_2 = secondary_trajectory[1:]
            velocity = (trajectory_2 - trajectory_1) / args.dt
            mid_trajectory = (trajectory_1 + trajectory_2) / 2

            v_flat = velocity.reshape(-1, D)
            mid_flat = mid_trajectory.reshape(-1, D)
            speed_square = metric_secondary.speed(mid_flat, v_flat)
            speed_square = speed_square.reshape(T-1, B, 1)
            secondary_energy_loss = speed_square.sum(dim=0).mean() * args.dt
            results['sync_secondary_energy_loss'] = secondary_energy_loss



        if args.time_plot:
            toc(t0, "Secondary Energy Loss")
        


        curr_sec_sub = z_curr_sec_sub
        next_sec_sub = z_next_sec_sub




        if args.time_plot:
            t0 = tic()
        # sync mmd loss
        if args.sync_mmd_loss:
            sync_mmd_loss = mmd_criterion(secondary_trajectory[-1], next_sec_sub)
            results['sync_mmd_loss'] = sync_mmd_loss


        if args.sync_pdf_loss:
            if args.unbalancedModel:
                sync_pdf_loss = calculate_pdf_loss(
                    secondary_trajectory[-1], next_sec_sub,
                    sinkhorn_loss=sinkhorn_secondary_list[t_idx + 1],
                    lnw_source=lnw_t[-1],
                    unbalancedModel=args.unbalancedModel,
                )
            else:
                sync_pdf_loss = calculate_pdf_loss(
                    secondary_trajectory[-1], next_sec_sub,
                    sinkhorn_loss=sinkhorn_secondary_list[t_idx + 1],
                )
            results['sync_pdf_loss'] = sync_pdf_loss
        if args.time_plot:
            toc(t0, "Sync PDF Loss")

        if args.time_plot:
            t0 = tic()  
            
        if args.sync_density_loss:
            sync_manifold_loss = 0
            
            
            secondary_trajectory_flatten = secondary_trajectory.reshape(-1, secondary_trajectory.shape[-1])
            sync_manifold_loss = sync_manifold_loss + calculate_density_loss(
                secondary_trajectory_flatten, 
                secondary_manifold_concat, 
                k=args.density_topk, 
                hinge_value=args.density_hinge
            )
            
            
            
            
            sync_manifold_loss = sync_manifold_loss + calculate_density_loss(
                secondary_trajectory[-1], 
                next_sec_sub, 
                k=args.density_topk, 
                hinge_value=args.density_hinge
            ) * args.terminal_density_weight
            results['sync_density_loss'] = sync_manifold_loss

        if args.time_plot:
            toc(t0, "Sync Manifold Density Loss")

    if args.accumulate_training:
        previous_sample['primal_x'] = zt_primal_sample[-1].detach()
        if args.unbalancedModel:
            previous_sample['primal_lnw'] = lnw_t[-1].detach()
        else:
            previous_sample['primal_lnw'] = None
        if args.sync_loss:
            #previous_sec_sample = secondary_trajectory[-1].clone().detach()
            previous_sample['secondary_x'] = next_sec_sub.detach()
        else:
            previous_sample['secondary_x'] = None
    else:
        previous_sample['primal_x'] = None
        previous_sample['primal_lnw'] = None
        previous_sample['secondary_x'] = None
    return results, previous_sample
