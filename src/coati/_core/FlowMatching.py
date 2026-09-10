import torch
import numpy as np
import ot as pot
from tqdm import tqdm
from typing import Optional
from typing import Union


def compute_pairing_uot_plans(
    X: list[np.ndarray], 
    t_train: torch.Tensor, 
    use_mini_batch: bool = False, 
    chunk_size: int = 1000,
    alpha_regm: float = 1.0,
    reg_strategy: str = "max_over_time",  # Options: "per_time" (per time-step) or "max_over_time" (global max)
    device : torch.device = torch.device("cpu")
) -> tuple[list[np.ndarray], list[Optional[dict]]]:
    """
    Compute Unbalanced Optimal Transport (UOT) plans between consecutive time steps.
    Supports two regularization strategies and mini-batch (chunked) computation.
    
    Args:
        X: List of numpy arrays, each representing data at a time step (shape [n_samples, ...]).
        t_train: Tensor of training time steps (shape [n_times]).
        use_mini_batch_uot: Whether to use mini-batch (chunked) UOT computation.
        chunk_size: Number of samples per chunk (for mini-batch mode).
        alpha_regm: Scaling factor for mass regularization (reg_m = alpha_regm * auto_tuned_reg_m).
        reg_strategy: Strategy for selecting regularization parameters:
            "per_time": Auto-tune reg/reg_m independently for each time step.
            "max_over_time": Auto-tune for all time steps, then use global max reg/reg_m.
    
    Returns:
        Tuple of (uot_plans, sampling_info_plans):
            uot_plans: List of UOT plan matrices (each shape [n_samples0, n_samples1]).
            sampling_info_plans: List of sampling info dicts (None for full-matrix mode).
    """
    # Validate regularization strategy
    if reg_strategy not in ["per_time", "max_over_time"]:
        raise ValueError(f"reg_strategy must be 'per_time' or 'max_over_time', got {reg_strategy}")
    
    uot_plans = []
    sampling_info_plans = []
    global_reg = None  # Global entropic reg (for "max_over_time" strategy)
    global_reg_m = None  # Global mass reg (for "max_over_time" strategy)

    # ------------------------------------------------------------------
    # Precompute global regularization parameters (for "max_over_time" strategy)
    # ------------------------------------------------------------------
    if reg_strategy == "max_over_time":
        reg_list = []
        reg_m_list = []
        print("Precomputing regularization parameters for all time steps...")
        
        # Auto-tune reg/reg_m for each time step to find global max
        for i in range(len(t_train) - 1):
            X_source, X_target = X[i], X[i + 1]
            n_source, n_target = X_source.shape[0], X_target.shape[0]
            a = np.ones(n_source)
            b = np.ones(n_target)

            # Compute and normalize cost matrix
            cost_matrix = pot.dist(X_source, X_target)
            if cost_matrix.mean() > 100:
                cost_matrix = cost_matrix / cost_matrix.max()

            # Use chunk data for mini-batch mode (match actual computation logic)
            if use_mini_batch:
                # Shuffle and split into chunks (same as mini-batch computation)
                n_chunks = n_source // chunk_size + 1
                source_perm = np.arange(n_source)
                np.random.shuffle(source_perm)
                target_perm = np.arange(n_target)
                np.random.shuffle(target_perm)
                source_chunks = np.array_split(source_perm, n_chunks)
                target_chunks = np.array_split(target_perm, n_chunks)
                
                # Use first chunk to compute reg/reg_m (consistent with mini-batch mode)
                first_src_chunk = source_chunks[0]
                first_tgt_chunk = target_chunks[0]
                sub_cost_matrix = cost_matrix[np.ix_(first_src_chunk, first_tgt_chunk)]
                sub_a = a[first_src_chunk]
                sub_b = b[first_tgt_chunk]
                
                reg, reg_m = calculate_auto_regularization(sub_a, sub_b, sub_cost_matrix)
                print(f"Time step {i} (mini-batch first chunk): reg={reg}, reg_m={reg_m}")
            else:
                # Use full matrix for reg/reg_m computation
                reg, reg_m = calculate_auto_regularization(a, b, cost_matrix)
                print(f"Time step {i} (full matrix): reg={reg}, reg_m={reg_m}")

            # Collect reg/reg_m for global max calculation
            reg_list.append(reg)
            reg_m_list.append(reg_m)

        # Set global parameters to the maximum of all time-step values
        global_reg = max(reg_list)
        global_reg_m = max(reg_m_list)
        print(f"Global max regularization parameters: reg={global_reg}, reg_m={global_reg_m}")

    # ------------------------------------------------------------------
    # Compute UOT plans for each time-step pair
    # ------------------------------------------------------------------
    for i in tqdm(range(len(t_train) - 1), desc="Computing UOT plans..."):
        X_source, X_target = X[i], X[i + 1]
        n_source, n_target = X_source.shape[0], X_target.shape[0]
        X_source = X_source.detach().cpu().numpy()
        X_target = X_target.detach().cpu().numpy()
        a = np.ones(n_source)
        b = np.ones(n_target)

        # Compute and normalize cost matrix
        cost_matrix = pot.dist(X_source, X_target)
        if cost_matrix.mean() > 100:
            cost_matrix = cost_matrix / cost_matrix.max()

        # Determine regularization parameters for current time step
        if reg_strategy == "per_time":
            # Auto-tune independently for each time step
            if not use_mini_batch:
                reg, reg_m = calculate_auto_regularization(a, b, cost_matrix)
                print(f"Time step {i} (per_time strategy): reg={reg}, reg_m={reg_m}")
            else:
                # Mini-batch mode: auto-tune later (using first chunk)
                reg, reg_m = None, None
        else:
            # Use precomputed global max parameters
            reg, reg_m = global_reg, global_reg_m
            print(f"Time step {i} (max_over_time strategy): using reg={reg}, reg_m={reg_m}")

        # ------------------------------
        # Full-matrix UOT computation
        # ------------------------------
        if not use_mini_batch:
            # Scale mass regularization with alpha_regm
            scaled_reg_m = reg_m * alpha_regm
            print(f"Time step {i}: scaled reg_m = {scaled_reg_m}")
            
            # Convert to CUDA for faster computation
            a_cuda = torch.from_numpy(a).float().to(device)
            b_cuda = torch.from_numpy(b).float().to(device)
            cost_matrix_cuda = torch.from_numpy(cost_matrix).float().to(device)

            # Compute UOT plan
            G = pot.unbalanced.sinkhorn_unbalanced(
                a_cuda, b_cuda, cost_matrix_cuda, reg, [scaled_reg_m, np.inf]
            )
            G = G.cpu().numpy()

            # Validate marginal constraints
            assert (np.abs(G.sum(axis=0) - b) < 1).all(), "UOT plan fails target marginal constraints"
            sampling_info_plans.append(None)

        # ------------------------------
        # Mini-batch UOT computation
        # ------------------------------
        else:
            # Calculate number of chunks
            n_chunks = n_source // chunk_size + 1
            
            # Initialize full UOT plan matrix
            G = np.zeros((n_source, n_target))
            
            # Shuffle and split indices into chunks
            source_perm = np.arange(n_source)
            np.random.shuffle(source_perm)
            target_perm = np.arange(n_target)
            np.random.shuffle(target_perm)
            source_chunks = np.array_split(source_perm, n_chunks)
            target_chunks = np.array_split(target_perm, n_chunks)

            uot_sub_plans = []  # Store chunk-specific UOT plans

            # Process each chunk pair
            for src_chunk, tgt_chunk in zip(source_chunks, target_chunks):
                # Extract chunk data
                sub_cost_matrix = cost_matrix[np.ix_(src_chunk, tgt_chunk)]
                sub_a = a[src_chunk]
                sub_b = b[tgt_chunk]

                # Auto-tune reg/reg_m using the first chunk (for "per_time" strategy)
                if reg_strategy == "per_time" and len(uot_sub_plans) == 0:
                    reg, reg_m = calculate_auto_regularization(sub_a, sub_b, sub_cost_matrix)
                    print(f"Time step {i} chunk 0 (per_time): reg={reg}, reg_m={reg_m}")

                # Scale mass regularization
                scaled_reg_m = reg_m * alpha_regm
                print(f"Time step {i} chunk: scaled reg_m = {scaled_reg_m}")

                # Convert chunk data to CUDA
                sub_a_cuda = torch.from_numpy(sub_a).float().to(device)
                sub_b_cuda = torch.from_numpy(sub_b).float().to(device)
                sub_cost_matrix_cuda =torch.from_numpy(sub_cost_matrix).float().to(device)

                # Compute UOT plan for the chunk
                G_sub = pot.unbalanced.sinkhorn_unbalanced(
                    sub_a_cuda, sub_b_cuda, sub_cost_matrix_cuda, reg, [scaled_reg_m, np.inf]
                )
                G_sub = G_sub.cpu().numpy()

                # Assign chunk plan to full matrix
                G[np.ix_(src_chunk, tgt_chunk)] = G_sub

                # Validate chunk marginal constraints
                assert (np.abs(G_sub.sum(axis=0) - sub_b) < 0.1 * chunk_size).all(), \
                    "Chunk UOT plan fails target marginal constraints"
                
                uot_sub_plans.append(G_sub.astype(np.float32))

            # Store sampling info for mini-batch mode
            sampling_info = {
                'sub_plans': uot_sub_plans,
                'source_groups': source_chunks,
                'target_groups': target_chunks
            }
            sampling_info_plans.append(sampling_info)

        uot_plans.append(G)

    return uot_plans, sampling_info_plans

def auto_tune_entropy_reg(C: np.ndarray) -> float:
    """
    Heuristic entropy regularization selection for balanced Sinkhorn.
    NumPy ONLY.
    """
    assert isinstance(C, np.ndarray)

    C_med = np.median(C)
    return max(1e-2, 0.1 * C_med)

def pad_t_like_x(t: Union[float, int, torch.Tensor], x: torch.Tensor) -> Union[float, int, torch.Tensor]:
    """
    Reshape time tensor `t` to match the dimensionality of data tensor `x` (for broadcasting).
    
    Args:
        t: Time value(s) (scalar or tensor).
        x: Data tensor, shape [batch_size, ...].
    
    Returns:
        Reshaped `t` with shape [batch_size, 1, ..., 1] (matches `x`'s batch dim and trailing dims).
    """
    if isinstance(t, (float, int)):
        return t
    # Add trailing singleton dimensions to match x's shape (after batch dim)
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """
    Base class for Conditional Flow Matching (CFM).
    Models the flow between source (x0) and target (x1) distributions at arbitrary time steps t.
    """
    def __init__(self, sigma: Union[float, int] = 0.0):
        """
        Initialize the ConditionalFlowMatcher.
        
        Args:
            sigma: Noise scale for sampling intermediate time-step data (xt).
        """
        self.sigma = sigma

    def compute_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Compute the mean of the intermediate distribution p(xt | x0, x1) at time t.
        Default: Linear interpolation between x0 and x1.
        
        Args:
            x0: Source data tensor, shape [batch_size, ...].
            x1: Target data tensor, shape [batch_size, ...].
            t: Time step(s) (scalar or tensor with shape [batch_size]).
        
        Returns:
            Mean tensor mu_t, shape [batch_size, ...].
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Compute the standard deviation of the intermediate distribution p(xt | x0, x1) at time t.
        Default: Constant sigma (independent of t).
        
        Args:
            t: Time step(s) (scalar or tensor).
        
        Returns:
            Standard deviation sigma_t (scalar or tensor).
        """
        del t  # Unused in base class
        return self.sigma

    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], 
                 epsilon: torch.Tensor) -> torch.Tensor:
        """
        Sample intermediate data xt from p(xt | x0, x1) using reparameterization.
        
        Args:
            x0: Source data tensor, shape [batch_size, ...].
            x1: Target data tensor, shape [batch_size, ...].
            t: Time step(s), shape [batch_size].
            epsilon: Noise tensor (from standard normal), shape [batch_size, ...].
        
        Returns:
            Sampled xt tensor, shape [batch_size, ...].
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)  # Match shape for broadcasting
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], 
                                 xt: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional flow (ut) at intermediate time t and data xt.
        Default: Constant flow (x1 - x0).
        
        Args:
            x0: Source data tensor, shape [batch_size, ...].
            x1: Target data tensor, shape [batch_size, ...].
            t: Time step(s), shape [batch_size].
            xt: Intermediate data tensor, shape [batch_size, ...].
        
        Returns:
            Conditional flow ut tensor, shape [batch_size, ...].
        """
        del t, xt  # Unused in base class
        return x1 - x0

    def sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        """Sample standard normal noise with the same shape as x."""
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, 
                                           t: Optional[torch.Tensor] = None, return_noise: bool = False) -> Union[tuple, tuple]:
        """
        Sample time steps t, intermediate data xt, and corresponding conditional flow ut.
        
        Args:
            x0: Source data tensor, shape [batch_size, ...].
            x1: Target data tensor, shape [batch_size, ...].
            t: Predefined time steps (optional). If None, samples t ~ Uniform(0,1).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            If return_noise: (t, xt, ut, epsilon)
            Else: (t, xt, ut)
        """
        # Sample uniform time steps if not provided
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t must have the same batch size as x0"

        # Sample noise and intermediate data
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        
        # Compute conditional flow
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute the lambda term for flow normalization (depends on sigma_t)."""
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma ** 2 + 1e-8)

def get_batch_ot_fm(
    FM: ConditionalFlowMatcher,
    X: list[np.ndarray],
    t_train: torch.Tensor,
    batch_size: int,
    ot_plans: list[np.ndarray],
    sampling_info_plans: list[Optional[dict]],
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct a training batch for balanced-OT-based Conditional Flow Matching (OT-CFM).

    Returns:
        t:  Sampled global time steps, shape [N]
        xt: Intermediate data, shape [N, ...]
        ut: Target conditional flow, shape [N, ...]
        eps: Noise used to generate xt, shape [N, ...]
    """

    ts = []
    xts = []
    uts = []
    noises = []

    # Iterate over consecutive time intervals
    for time_idx in range(len(t_train) - 1):
        ot_plan = ot_plans[time_idx]
        sampling_info = sampling_info_plans[time_idx]

        x0_np = X[time_idx]
        x1_np = X[time_idx + 1]

        # --------------------------------------------------
        # Sample paired (x0, x1) using OT plan
        # --------------------------------------------------
        sampled_x0_np, sampled_x1_np, _, _ = sample_from_ot_plan(
            ot_plan=ot_plan,
            x0=x0_np,
            x1=x1_np,
            batch_size=batch_size,
            sampling_info=sampling_info,
        )

        if sampled_x0_np.size == 0:
            continue

        x0 = sampled_x0_np.float().to(device)
        x1 = sampled_x1_np.float().to(device)

        # --------------------------------------------------
        # Sample CFM quantities in local time [0, 1]
        # --------------------------------------------------
        t_local, xt, ut, eps = FM.sample_location_and_conditional_flow(
            x0=x0,
            x1=x1,
            return_noise=True,
        )

        # --------------------------------------------------
        # Map local time to global time axis
        # --------------------------------------------------
        delta_t = (t_train[time_idx + 1] - t_train[time_idx]).to(device)
        t_global = t_train[time_idx].to(device) + t_local * delta_t

        # --------------------------------------------------
        # Normalize flow to per-unit-time
        # --------------------------------------------------
        ut = ut / delta_t

        # --------------------------------------------------
        # Accumulate batch
        # --------------------------------------------------
        ts.append(t_global)
        xts.append(xt)
        uts.append(ut)
        noises.append(eps)

    # ------------------------------------------------------
    # Concatenate across time intervals
    # ------------------------------------------------------
    if not ts:
        return (
            torch.empty(0, device=device),
            torch.empty(0, *X[0].shape[1:], device=device),
            torch.empty(0, *X[0].shape[1:], device=device),
            torch.empty(0, *X[0].shape[1:], device=device),
        )

    return (
        torch.cat(ts, dim=0),
        torch.cat(xts, dim=0),
        torch.cat(uts, dim=0),
        torch.cat(noises, dim=0),
    )

def sample_from_ot_plan(
        ot_plan: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
        batch_size: int,
        sampling_info: Optional[dict] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample matching (x0, x1) pairs from an OT plan.
    Supports both full-matrix and mini-batch (chunked) OT plans.
    
    Args:
        ot_plan: OT plan matrix (full: [n_samples0, n_samples1]; chunked: aggregated from sub-plans).
        x0: Source dataset array, shape [n_samples0, ...].
        x1: Target dataset array, shape [n_samples1, ...].
        batch_size: Number of pairs to sample.
        sampling_info: Sampling info dict for mini-batch mode (contains chunk indices and sub-plans).
    
    Returns:
        Tuple of (sampled_x0, sampled_x1, sampled_i, sampled_j):
            sampled_x0: Sampled source data, shape [batch_size, ...].
            sampled_x1: Sampled target data, shape [batch_size, ...].
            sampled_i: Source indices of sampled pairs, shape [batch_size].
            sampled_j: Target indices of sampled pairs, shape [batch_size].
            Empty arrays if OT plan has insufficient mass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Sampling from full OT plan
    # ------------------------------
    if sampling_info is None:
        # Convert OT plan to CUDA tensor
        pi_cuda = torch.from_numpy(ot_plan.astype(np.float32)).to(device)
        
        # Compute row sums (source-side mass) and total mass
        row_sums = pi_cuda.sum(axis=1)
        total_mass = row_sums.sum()
        
        # Return empty if total mass is too small (numerical failure)
        if total_mass < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Sample source indices (proportional to row sums)
        row_probs = row_sums / total_mass
        sampled_i = torch.multinomial(row_probs, num_samples=batch_size, replacement=True)
        
        # Sample target indices (conditional on source indices)
        selected_rows = pi_cuda[sampled_i]
        selected_row_sums = row_sums[sampled_i]
        conditional_probs = selected_rows / (selected_row_sums.unsqueeze(1) + 1e-12)  # Avoid division by zero
        sampled_j = torch.multinomial(conditional_probs, num_samples=1).squeeze(1)
        
        # Convert indices to numpy
        sampled_i_np = sampled_i.cpu().numpy()
        sampled_j_np = sampled_j.cpu().numpy()

    # ------------------------------
    # Sampling from mini-batch OT plan
    # ------------------------------
    else:
        # Extract chunk info from sampling_info
        sub_plans = sampling_info['sub_plans']
        source_chunks = sampling_info['source_groups']
        target_chunks = sampling_info['target_groups']
        
        # Compute chunk masses and total mass
        chunk_masses = [sub_plan.sum() for sub_plan in sub_plans]
        total_mass = sum(chunk_masses)
        
        # Return empty if total mass is too small
        if total_mass < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Sample chunks (proportional to chunk masses)
        chunk_probs = torch.tensor(chunk_masses, dtype=torch.float32, device=device) / total_mass
        sampled_chunk_indices = torch.multinomial(chunk_probs, num_samples=batch_size, replacement=True)
        
        # Convert chunk data to CUDA for faster processing
        sub_plans_cuda = [torch.from_numpy(sp).to(device) for sp in sub_plans]
        source_chunks_cuda = [torch.from_numpy(sc).to(device) for sc in source_chunks]
        target_chunks_cuda = [torch.from_numpy(tc).to(device) for tc in target_chunks]
        
        # Get unique chunks and their sample counts
        unique_chunks, chunk_counts = torch.unique(sampled_chunk_indices, return_counts=True)
        
        # Initialize arrays for sampled indices
        sampled_i_cuda = torch.empty(batch_size, dtype=torch.long, device=device)
        sampled_j_cuda = torch.empty(batch_size, dtype=torch.long, device=device)

        # Process each unique chunk
        for chunk_idx, count in zip(unique_chunks, chunk_counts):
            # Get current chunk's OT plan and indices
            chunk_plan = sub_plans_cuda[chunk_idx]
            chunk_source_idx = source_chunks_cuda[chunk_idx]
            chunk_target_idx = target_chunks_cuda[chunk_idx]
            
            # Compute row sums (source-side mass) for the chunk
            chunk_row_sums = chunk_plan.sum(axis=1)
            chunk_total_mass = chunk_row_sums.sum()
            
            # Skip if chunk has insufficient mass
            if chunk_total_mass < 1e-9:
                continue
            
            # Sample source indices within the chunk
            chunk_row_probs = chunk_row_sums / chunk_total_mass
            local_source_idx = torch.multinomial(chunk_row_probs, num_samples=count.item(), replacement=True)
            
            # Sample target indices within the chunk (conditional on source)
            selected_chunk_rows = chunk_plan[local_source_idx]
            selected_chunk_row_sums = chunk_row_sums[local_source_idx]
            chunk_conditional_probs = selected_chunk_rows / (selected_chunk_row_sums.unsqueeze(1) + 1e-12)
            local_target_idx = torch.multinomial(chunk_conditional_probs, num_samples=1).squeeze(1)
            
            # Map local chunk indices to global dataset indices
            global_source_idx = chunk_source_idx[local_source_idx]
            global_target_idx = chunk_target_idx[local_target_idx]
            
            # Assign to final index arrays (using mask for chunk position)
            chunk_mask = (sampled_chunk_indices == chunk_idx)
            sampled_i_cuda[chunk_mask] = global_source_idx.long()
            sampled_j_cuda[chunk_mask] = global_target_idx.long()
        
        # Convert CUDA indices to numpy
        sampled_i_np = sampled_i_cuda.cpu().numpy()
        sampled_j_np = sampled_j_cuda.cpu().numpy()

    # Return empty arrays if no valid indices were sampled
    if sampled_i_np.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Extract sampled data pairs using indices
    sampled_x0 = x0[sampled_i_np]
    sampled_x1 = x1[sampled_j_np]
    
    return sampled_x0, sampled_x1, sampled_i_np, sampled_j_np


import warnings
def calculate_auto_regularization(a: np.ndarray, b: np.ndarray, M: np.ndarray, tolerance: float = 1e-3, device = 'cpu') -> tuple[float, float]:
    """
    Two-step automatic tuning for unbalanced Sinkhorn regularization parameters:
    Step 1: Select entropic regularization (reg) using the elbow rule on transport cost ⟨G,M⟩.
    Step 2: Select mass regularization (reg_m) via grid search + elbow rule on transport cost.
    
    Args:
        a: Source marginal distribution, shape [n_samples0].
        b: Target marginal distribution, shape [n_samples1].
        M: Cost matrix, shape [n_samples0, n_samples1].
        tolerance: Tolerance for numerical stability checks (unused).
    
    Returns:
        reg: Optimal entropic regularization strength.
        reg_m: Optimal mass regularization strength.
    """
    # Fallback to defaults if cost matrix is empty
    if M.size == 0:
        print("Cost matrix is empty – falling back to default parameters.")
        return 1e-5, 1.0
    

    # ------------------------------------------------------------------
    # Step 1: Select entropic regularization (reg) via elbow rule
    # ------------------------------------------------------------------
    reg_list, loss_list = [], []  # For logging (unused in final selection)
    fixed_reg_m = 50.0  # Fixed mass regularization for reg selection
    reg = 10.0  # Default fallback value

    # Context manager to convert specific warnings to exceptions (for stability checks)
    class catch_specific_warning:
        def __init__(self, message: str, category: Warning, module: str):
            self.message = message
            self.category = category
            self.module = module

        def __enter__(self):
            self.original_filters = warnings.filters.copy()
            warnings.filterwarnings(
                "error",
                message=self.message,
                category=self.category,
                module=self.module
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            warnings.filters = self.original_filters
            return False  # Do not suppress exceptions

    # Check if OT plan is numerically stable (finite values, positive mass, non-zero sum)
    def is_stable(ot_plan: np.ndarray, a: np.ndarray) -> bool:
        return (
            np.all(np.isfinite(ot_plan)) and
            np.sum(ot_plan) > 1e-8 and
            np.all(ot_plan >= 0)
        )

    # Round 1: Coarse search for stable reg (step size = 2e-2)
    eps_min, eps_step, eps_max = 5e-2, 2e-2, 10.0
    current_eps = eps_min
    first_valid_eps = None

    while current_eps <= eps_max:
        try:
            # Catch numerical errors from unbalanced Sinkhorn
            with catch_specific_warning(
                    message="Numerical errors at iteration.*",
                    category=UserWarning,
                    module="ot.unbalanced._sinkhorn"):
                # Compute unbalanced OT plan
                ot_plan = pot.unbalanced.sinkhorn_unbalanced(
                    a=a,
                    b=b,
                    M=M,
                    reg=current_eps,
                    reg_m=[fixed_reg_m, np.inf]
                )
                
                # Check stability of the OT plan
                if is_stable(ot_plan, a):
                    first_valid_eps = current_eps
                    break
        
        except (Exception, UserWarning) as exc:
            print(f"[Round-1 eps={current_eps:.3e}] Failed: {type(exc).__name__}: {exc}")
        
        # Move to next epsilon
        current_eps += eps_step

    # If no valid eps found in coarse search, use default
    if first_valid_eps is None:
        print("No stable eps found in coarse search – keeping default reg =", reg)
    else:
        # Round 2: Fine search for optimal reg (step size = 1e-3)
        fine_eps_min = first_valid_eps + 1e-3
        fine_eps_step = 1e-3
        current_eps = fine_eps_min
        best_eps = None

        while current_eps <= eps_max:
            try:
                with catch_specific_warning(
                        message="Numerical errors at iteration.*",
                        category=UserWarning,
                        module="ot.unbalanced._sinkhorn"):
                    ot_plan = pot.unbalanced.sinkhorn_unbalanced(
                        a=a,
                        b=b,
                        M=M,
                        reg=current_eps,
                        reg_m=[fixed_reg_m, np.inf]
                    )
                    
                    if is_stable(ot_plan, a):
                        best_eps = current_eps
                        break
            
            except (Exception, UserWarning) as exc:
                print(f"[Round-2 eps={current_eps:.3e}] Failed: {type(exc).__name__}: {exc}")
            
            current_eps += fine_eps_step

        # Update reg with fine search result (or coarse search result if fine search fails)
        reg = best_eps if best_eps is not None else first_valid_eps
        print(f"Final entropic reg selected: {reg}")

    # ------------------------------------------------------------------
    # Step 2: Select mass regularization (reg_m) via grid search + elbow rule
    # ------------------------------------------------------------------
    # Log-spaced grid of reg_m candidates (40 points from 1e-2 to 10^1.2)
    reg_m_candidates = np.logspace(-2, 1.2, 40)
    reg_m_list = []
    transport_loss_list = []

    # Evaluate each reg_m candidate
    for reg_m in reg_m_candidates:
        try:
            # Compute unbalanced OT plan with fixed reg (from Step 1)
            G = pot.unbalanced.sinkhorn_unbalanced(
                a=a,
                b=b,
                M=M,
                reg=reg,
                reg_m=[reg_m, np.inf]
            )
            
            # Skip if OT plan is unstable
            if not (np.all(np.isfinite(G)) and G.sum() > 1e-6 and G.min() >= 0):
                continue
            
            # Compute transport cost (⟨G, M⟩)
            transport_loss = float((G * M).sum())
            reg_m_list.append(reg_m)
            transport_loss_list.append(transport_loss)
        
        except Exception:
            continue

    # Handle case with too few valid reg_m candidates
    if len(reg_m_list) < 4:
        # Fallback to reg_m with minimum transport loss
        best_reg_m = reg_m_list[np.argmin(transport_loss_list)] if reg_m_list else 1.0
        print(f"Insufficient valid reg_m candidates – using min-loss reg_m: {best_reg_m}")
    else:
        # Convert to numpy arrays for processing
        x = np.array(reg_m_list, dtype=float)
        y = np.array(transport_loss_list, dtype=float)

        # Normalize x (reg_m) and y (transport loss) to [0, 1]
        x_norm = (x - x[0]) / (x[-1] - x[0] + 1e-12)  # Avoid division by zero
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

        # Define line from first to last point (baseline for elbow detection)
        y0, y1 = y_norm[0], y_norm[-1]
        line_y = y0 + (y1 - y0) * x_norm  # Y-values of the baseline line

        # Compute direction vector of the baseline line
        line_vec = np.array([1.0, y1 - y0])
        line_length = np.linalg.norm(line_vec)

        # Compute perpendicular distance from each point to the baseline
        # Cross product gives perpendicular distance (scaled by line length)
        distances = np.abs(np.cross(
            line_vec, 
            np.column_stack([x_norm - x_norm[0], y_norm - y_norm[0]])
        )) / line_length

        # Elbow is the point with maximum distance to the baseline
        best_reg_m = reg_m_list[np.argmax(distances)]
        print(f"Elbow rule selected reg_m: {best_reg_m:.6f}")

    return reg, best_reg_m