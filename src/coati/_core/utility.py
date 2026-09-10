import torch
import math
import os
import time
from .DataLoad import *
from torchdiffeq import odeint


def forward_ode(func, initial_distribution, device, unbalancedModel,
                viz_timesteps=30, t_start=0, t_end=3, method='rk4', atol=1e-5, rtol=1e-5):
    """
    Run forward ODE integration, supporting both balanced and unbalanced models.

    Args:
        func: neural ODE vector field
        device: torch device
        args: must have args.unbalancedModel (bool)
        viz_timesteps: number of integration steps
        t_end: end time
        method: ODE solver method
        atol, rtol: solver tolerances

    Returns:
        all_samples_forward: (T, N, D) tensor of trajectories
        lnw_t: (T, N, 1) log-weights (None if balanced)
    """
    time_points = torch.linspace(t_start, t_end, viz_timesteps).to(device)

    with torch.no_grad():
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        z0 = initial_distribution
        N = z0.shape[0]

        if unbalancedModel:
            lnw_0 = torch.log(torch.ones(N, 1) / N).to(device)
            e_v_0 = torch.zeros(N, 1).to(device)
            e_g_0 = torch.zeros(N, 1).to(device)
            all_samples_forward, lnw_t, e_v_t, e_g_t = odeint(
                func,
                (z0, lnw_0, e_v_0, e_g_0),
                time_points,
                atol=atol, rtol=rtol, method=method,
            )
            return all_samples_forward, lnw_t, e_v_t, e_g_t
        else:
            e_0 = torch.zeros(N, 1).to(device)
            all_samples_forward, e_t = odeint(
                func,
                (z0, e_0),
                time_points,
                atol=atol, rtol=rtol, method=method,
            )
        return all_samples_forward, e_t




def precompute_gaussian_params(point_cloud_a, sigma, device):
    """
    Compute GMM parameters where each point has its own Gaussian component.
    
    Args:
        point_cloud_a: Reference point cloud [N, D]
        sigma: Initial scale for covariance matrices
        device: torch device
    
    Returns:
        weights: [N, ] the weights of different components (equal weights)
        means: [N, D] each point becomes a component mean
        precisions: [N, D, D] the inverse of the covariance matrices
    """
    N, D = point_cloud_a.shape
    
    # Each point becomes a component with equal weight
    weights = torch.ones(N, device=device) / N
    
    for i in range(point_cloud_a.shape[0]):
        dists = torch.sum((point_cloud_a - point_cloud_a[i])**2, dim=1)

        neighbors = torch.sum(dists < (4 * sigma * sigma))
        weights[i] = 1.0 / (neighbors.float() + 1.0) 
        
    # normalize weights
    weights = weights / torch.sum(weights)
    
    # Each point is a component mean
    means = point_cloud_a.clone()
    
    adaptive_sigma = sigma * (1.0 + 0.1 * math.log(max(D, 1)))
    
    # Create precision matrices (inverse of covariance matrices)
    precision_scalar = 1.0 / (adaptive_sigma ** 2)
    precisions = precision_scalar * torch.eye(D, device=device).repeat(N, 1, 1)
    
    return weights, means, precisions

def save_checkpoint(func, optimizer, scheduler, itr, loss_meter, args, filename='checkpoint.pth'):
    """Save model checkpoint"""
    checkpoint = {
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'iteration': itr,
        'loss_meter': loss_meter.avg,
        'args': args
    }
    torch.save(checkpoint, os.path.join(args.train_dir, filename))

def load_checkpoint(func, args, ckpt_path='ckpt_latest.pth', train_dir=None, device='cpu', optimizer=None, scheduler=None):
    """Load model checkpoint, optionally restore optimizer and scheduler"""
    if train_dir is None:
        train_dir = args.train_dir
    ckpt_path = os.path.join(train_dir, ckpt_path)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        func.load_state_dict(checkpoint['func_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Optimizer state restored')
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'Scheduler state restored')
        start_itr = checkpoint['iteration']
        print(f'Loaded checkpoint from {ckpt_path}, iteration {start_itr}')
        return start_itr
    else:
        print(f'No checkpoint found at {ckpt_path}')
        return 1

class SectionTimer:
    def __init__(self, device):
        self.device = device
        self.times = {}

    def start(self, name):
        if self.device.type == 'mps':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.time()
        self.current = name

    def end(self):
        if self.device.type == 'mps':
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed = self.start_event.elapsed_time(self.end_event)  # ms
        else:
            elapsed = (time.time() - self.start_time) * 1000
        self.times[self.current] = self.times.get(self.current, 0) + elapsed

    def summary(self):
        return self.times
import random
import numpy as np


def set_global_seed(seed: int):
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU / MPS
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_exp_name(args):
    """
    Build a hyperparameter-encoded experiment name.

    The returned string is shared by:
      - checkpoint filenames (see Epoch.py)
      - TensorBoard log subdirectory (pass as part of SummaryWriter log_dir)
    so that sweeps over hyperparameters don't overwrite each other.

    Naming convention:
      no-sync: s{seed}_e{energy}_m{pdf}_d{density}
      sync:    s{seed}_e{energy}_m{pdf}_d{density}_a{sync_weight}

    where
      s = random seed
      e = energy_coefficient
      m = pdf_coefficient  (M for "marginal matching")
      d = density_coefficient
      a = sync_weight (alpha, only when sync_loss=True)

    Args:
        args: argparse Namespace / Args object with the fields above.
            `sync_weight` is expected only when `sync_loss` is True.

    Returns:
        str: experiment name, e.g. "s0_e0.1_m1.0_d0.01_a0.5"
    """
    parts = [
        f"s{args.seed}",
        f"e{args.energy_coefficient}",
        f"m{args.pdf_coefficient}",
        f"d{args.density_coefficient}",
    ]
    if getattr(args, "sync_loss", False):
        parts.append(f"a{args.sync_weight}")
    return "_".join(parts)

def log_losses_to_tensorboard(writer, Whole_results, itr, args):
    """
    Write losses to TensorBoard
    
    Args:
        writer: TensorBoard SummaryWriter object
        Whole_results: A dictionary containing all loss values
        itr: the current iteration number
        args
    """
    if writer is None:
        return
    
    # Helper function: safely convert tensor to scalar
    def to_scalar(value):
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else float(value)
        return float(value)

    primal_loss =  (Whole_results['forward_density_loss'] * args.density_coefficient 
    + Whole_results['forward_sinkhorn_pdf_loss'] * args.pdf_coefficient 
    + Whole_results['forward_energy_loss'] * args.energy_coefficient 
    + Whole_results['reverse_matching_loss'] * args.pdf_coefficient 
    + Whole_results['forward_mmd_loss'] * args.pdf_coefficient 
    +Whole_results['forward_mass_loss'] * args.mass_coefficient 
    )

    total_loss = primal_loss


    if args.sync_loss:
        sec_loss = ( Whole_results['sync_secondary_energy_loss'] * args.energy_coefficient 
        + Whole_results['sync_density_loss'] * args.density_coefficient)

        if args.sync_pdf_loss:
            sec_loss = sec_loss + Whole_results['sync_pdf_loss'] * args.pdf_coefficient 
        if args.sync_mmd_loss:
            sec_loss = sec_loss + Whole_results['sync_mmd_loss'] * args.pdf_coefficient

        total_loss = total_loss + sec_loss

    writer.add_scalar('Loss/total', to_scalar(total_loss), itr)
    writer.add_scalar('Primal/primal_total_loss', to_scalar(primal_loss), itr)
    writer.add_scalar('Primal/manifold_constraint_loss', to_scalar(Whole_results['forward_density_loss']), itr)
    writer.add_scalar('Primal/terminal_matching_loss', to_scalar(Whole_results['forward_sinkhorn_pdf_loss']), itr)
    
    if args.unbalancedModel:
        writer.add_scalar('Primal/mass_constraint', to_scalar(Whole_results['forward_mass_loss']), itr)
        writer.add_scalar('Primal/rhov2_unbalanced', to_scalar(Whole_results['forward_energy_velocity']), itr)
        writer.add_scalar('Primal/rhog2_unbalanced', to_scalar(Whole_results['forward_energy_growth']), itr)
    else:
        writer.add_scalar('Primal/rhov2_balanced', to_scalar(Whole_results['forward_energy_loss']), itr)
    if args.mmd_loss:
        writer.add_scalar('Primal/mmd_terminal_matching', to_scalar(Whole_results['forward_mmd_loss']), itr)
    if args.reverseMatching:
        writer.add_scalar('Primal/reverse_matching_loss', to_scalar(Whole_results['reverse_matching_loss']), itr)
    
    if args.sync_loss:
        writer.add_scalar('Secondary/sync_total_loss', to_scalar(sec_loss), itr)
        writer.add_scalar('Secondary/sync_rhov2', to_scalar(Whole_results['sync_secondary_energy_loss']), itr)
        if args.sync_pdf_loss:
            writer.add_scalar('Secondary/sync_terminal_matching', to_scalar(Whole_results['sync_pdf_loss']), itr)
        if args.sync_mmd_loss:
            writer.add_scalar('Secondary/sync_mmd_matching', to_scalar(Whole_results['sync_mmd_loss']), itr)
        if args.sync_density_loss:
            writer.add_scalar('Secondary/sync_manifold_constraint', to_scalar(Whole_results['sync_density_loss']), itr)
