import torch
import numpy as np
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Independent, Normal
from sklearn.neighbors import NearestNeighbors
def compute_adaptive_sigmas(points_np, base_sigma=0.01, k=30):
    """
    points_np: [N, D] numpy array
    output: sigma value for each point [N]
    """
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points_np)
    distances, _ = nbrs.kneighbors(points_np)
    avg_distance = distances[:, 1:].mean(axis=1)

    # Normalize distances to [0.1, 1.0] for numerical stability
    normalized = (avg_distance - avg_distance.min()) / (avg_distance.max() - avg_distance.min() + 1e-8)
    scaled = 0.001 + 9.999 * normalized  # scale to [0.0001, 10.0]
    sigmas = base_sigma * scaled

    return torch.tensor(sigmas, dtype=torch.float32)

'''
def create_mixture_gaussian(mu, base_sigma, adaptive_sigma, device):
    """
    mu: [N, D] tensor, point cloud
    base_sigma: float, control the overall variance scale
    
    return: mixture of gaussian: torch.distributions.MixtureSameFamily
    """
    num_components, dim = mu.shape
    
    # Step 1: adaptive sigma
    if adaptive_sigma:
        sigmas = compute_adaptive_sigmas(mu.cpu().numpy(), base_sigma=base_sigma).to(device)  # shape [N]
    else:
        sigmas = torch.ones(num_components, device=device) * base_sigma
        
    sigmas = sigmas
    mu = mu

    # Step 2: build N covariance matrices (sigma_i * I)
    cov_matrices = torch.stack([
        sigma_i * torch.eye(dim, device=device) for sigma_i in sigmas
    ])  # shape: [N, D, D]

    # Step 3: uniform weights 
    weights = torch.ones(num_components, device=device) / num_components
    categorical = Categorical(probs=weights)

    # Step 4: build Multivariate Normal and Mixture
    component_dist = MultivariateNormal(loc=mu.to(device), covariance_matrix=cov_matrices)
    mixture = MixtureSameFamily(mixture_distribution=categorical,
                                 component_distribution=component_dist)
    return mixture
'''
@torch.no_grad()
def _kmeans(mu, K, iters=20):
    # mu: [N, D]
    N, D = mu.shape
    # k-means++ initialization
    centers = mu[torch.randint(0, N, (1,))]
    d2 = torch.cdist(mu, centers).squeeze(-1).pow(2)
    for _ in range(K - 1):
        probs = d2 / (d2.sum() + 1e-12)
        idx = torch.multinomial(probs, 1)
        centers = torch.cat([centers, mu[idx]], dim=0)
        d2 = torch.minimum(d2, torch.cdist(mu, centers[-1:].contiguous()).squeeze(-1).pow(2))

    for _ in range(iters):
        # assignment
        dist2 = torch.cdist(mu, centers).pow(2)        # [N, K]
        assign = dist2.argmin(dim=1)                    # [N]
        # update centers
        new_centers = []
        for k in range(K):
            m = mu[assign == k]
            if m.numel() == 0:
                new_centers.append(centers[k:k+1])    
            else:
                new_centers.append(m.mean(dim=0, keepdim=True))
        new_centers = torch.cat(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers
    return centers, assign


def create_mixture_gaussian(mu, base_sigma, adaptive_sigma, device, max_components=200, kmeans_iters=20):
    """
    mu: [N, D] point cloud
    base_sigma: float
    adaptive_sigma: bool
    max_components: compressed components number K (e.g. 100)
    return: MixtureSameFamily
    """
    mu = mu.detach().to(device)
    N, D = mu.shape
    K = min(max_components, N)


    if adaptive_sigma:

        sigmas = compute_adaptive_sigmas(mu.cpu().numpy(), base_sigma=base_sigma)
        sigmas = torch.as_tensor(sigmas, device=device, dtype=mu.dtype)  # [N]
    else:
        sigmas = torch.full((N,), base_sigma, device=device, dtype=mu.dtype)

    # KMeans compression
    centers, assign = _kmeans(mu, K, iters=kmeans_iters)                 # centers: [K, D]

    # each cluster weight (by cluster size)
    counts = torch.bincount(assign, minlength=K).clamp_min(1)            # [K]
    weights = counts.float() / float(N)

    # each cluster variance (isotropic): cluster intra-dispersion 
    # cluster intra-variance can be used as isotropic variance by averaging over all dimensions
    cluster_var = torch.empty(K, device=device, dtype=mu.dtype)
    for k in range(K):
        idx = (assign == k)
        pts = mu[idx]
        if pts.shape[0] <= 1:
            cluster_var[k] = 0.0
        else:
            # each dimension sample variance -> average over all dimensions as isotropic variance
            v = pts.var(dim=0, unbiased=False).mean()
            cluster_var[k] = v

    # mean of original sigmas (当作方差) in cluster, to preserve adaptive scale information
    sigma_intra = torch.zeros(K, device=device, dtype=mu.dtype)
    for k in range(K):
        idx = (assign == k)
        if idx.any():
            sigma_intra[k] = sigmas[idx].mean()
        else:
            sigma_intra[k] = base_sigma

    # combine final variance (can adjust weights by demand, below is simple addition)
    final_var = (cluster_var + sigma_intra).clamp_min(1e-12)             # [K]
    std = final_var.sqrt().unsqueeze(-1)                                 # [K, 1]

    # component distribution (isotropic diagonal)
    component = Independent(Normal(loc=centers, scale=std), 
                            reinterpreted_batch_ndims=1)
    # mixture weights
    categorical = Categorical(probs=weights)

    return MixtureSameFamily(categorical, component)



def process_temporal_data(primary_data, time_labels=None):
    """
    process temporal data, including normalization and splitting
    
    Args:
        primary_data (list of torch.Tensor): original data list, each element is a time point data
        time_labels (list of str, optional): time point labels list
    
    Returns:
        tuple: (processed data list, number of samples for each time point)
    """
    # calculate the number of samples for each time point
    n_samples = [data.shape[0] for data in primary_data]
    
    # merge all data
    total_data = torch.cat(primary_data, dim=0)
    
    # normalize to unit cube
    total_normalized, _ = normalize_to_unit_cube(total_data)
    
    # split normalized data
    normalized_data_list = []
    start_idx = 0
    for n in n_samples:
        end_idx = start_idx + n
        normalized_data_list.append(total_normalized[start_idx:end_idx])
        start_idx = end_idx
    
    # print information
    print(f"Number of time points: {len(time_labels) if time_labels else len(primary_data)}")
    for i, (n, data) in enumerate(zip(n_samples, normalized_data_list)):
        time_label = time_labels[i] if time_labels else f't{i}'
        print(f"{time_label}: {n} samples, shape: {data.shape}")
    
    return normalized_data_list, n_samples

def load_source_data(data_path, time_labels, support_points, DataShape=True, device=None):
    """
    load data from data_path
    
    Args:
        data_path: data path
        time_labels: list of time labels
        device: computation device, default is None (use CPU)
    
    Returns:
        data_tensors: list of tensors, each tensor is a time point data
    """
    # set computation device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load data
    try:
        data = np.load(data_path, allow_pickle=True)
        print(f"Data type: {type(data)}")
        
        # check if data is a NpzFile object
        if isinstance(data, np.lib.npyio.NpzFile) or isinstance(data, np.lib.npyio.NpyFile):
            print(f"Available keys in npz file: {list(data.keys())}")
            # process .npz file
            data_tensors = []
            if support_points:
                for i, label in enumerate(time_labels[:-1]):
                    if label in data:
                        arr = torch.from_numpy(data[label]).type(torch.float32).to(device)
                        data_tensors.append(arr)
                        if DataShape:
                            print(f"time point {label} data shape: {arr.shape}")
                    else:
                        print(f"Warning: key '{label}' not found in npz file")
                        
                support_tensors = []
                arr = torch.from_numpy(data[time_labels[-1]]).type(torch.float32).to(device)
                support_tensors.append(arr)
                if DataShape:
                    print(f"support points data shape: {arr.shape}")
                return data_tensors, support_tensors
                
            else:
                for i, label in enumerate(time_labels):
                    if label in data:
                        arr = torch.from_numpy(data[label]).type(torch.float32).to(device)
                        data_tensors.append(arr)
                        if DataShape:
                            print(f"time point {label} data shape: {arr.shape}")
                    else:
                        print(f"Warning: key '{label}' not found in npz file")
                        
                return data_tensors
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def normalize_to_unit_cube(data, target_std=0.12, target_center=0.5, save_path='norm_params.pt'):
    """
    Normalize data to fit within a unit cube with specified standard deviation and center
    
    Args:
        data (torch.Tensor): Input data of shape (n_samples, n_features)
        target_std (float): Target standard deviation for the normalized data
        target_center (float or torch.Tensor): Target center for the normalized data
                                              If float, same center for all dimensions
                                              If tensor, specific center for each dimension
    
    Returns:
        tuple: (normalized_data, norm_params)
            - normalized_data (torch.Tensor): Normalized data
            - norm_params (dict): Dictionary containing normalization parameters:
                - mean: Original data mean
                - std: Original data standard deviation
                - target_std: Target standard deviation
                - target_center: Target center
                - shift: Final shift applied
    """
    # Convert to torch tensor if not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Get device
    device = data.device
    
    # Calculate mean and std
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    
    # Initial normalization
    normalized_data = (data - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Rescale to target standard deviation
    normalized_data *= target_std
    
    # Calculate current center and shift to target center
    min_vals, _ = normalized_data.min(dim=0)
    max_vals, _ = normalized_data.max(dim=0)
    current_center = (min_vals + max_vals) / 2.0
    
    # Handle target_center
    if isinstance(target_center, (int, float)):
        target_center = torch.full((data.shape[1],), target_center, device=device)
    else:
        target_center = torch.tensor(target_center, device=device)
    
    # Calculate and apply shift
    shift = target_center - current_center
    normalized_data += shift
    
    # Store normalization parameters
    norm_params = {
        'mean': mean,
        'std': std,
        'target_std': target_std,
        'target_center': target_center,
        'shift': shift
    }
    
    torch.save(norm_params, save_path)
    
    return normalized_data, norm_params

def denormalize_from_unit_cube(normalized_data, norm_params):
    """
    Convert normalized data back to original space
    
    Args:
        normalized_data (torch.Tensor): Normalized data
        norm_params (dict): Dictionary containing normalization parameters
    
    Returns:
        torch.Tensor: Data in original space
    """
    if norm_params is None:
        norm_params = torch.load('norm_params.pt')
        
    # Remove the shift
    data = normalized_data - norm_params['shift']
    
    # Remove target_std scaling
    data = data / norm_params['target_std']
    
    # Convert back to original space
    data = data * (norm_params['std'] + 1e-12) + norm_params['mean']
    
    return data

def Sampling(num_samples, device, source_data, sigma=0.0001, rare_files = None, rare_factor=300.0):
    """
    Sample points from source data with Gaussian noise
    
    Args:
        num_samples: Number of points to sample
        device: Computation device
        source_data: Source data to sample from
        sigma: Standard deviation for Gaussian noise (default: 0.02)
    
    Returns:
        source_sample: Sampled points with noise
        logp_diff_t1: Terminal log probability difference (zeros)
    """
    # Get data dimensions
    dim = source_data.shape[1]
    num_source_points = len(source_data)
    
    weights = np.ones(num_source_points, dtype=np.float32)
    
    rare_indices = set()
    if rare_files is not None:
        for file in rare_files:
            with open(file, "r") as f:
                content = f.read().strip()
                if content:
                    indices = list(map(int, content.split(",")))
                    rare_indices.update(indices)
        for idx in rare_indices:
            if 0 <= idx < num_source_points:
                weights[idx] *= rare_factor
                
    probs = weights / weights.sum()
    
    # Sample indices (with replacement if num_samples > num_source_points)
    if num_samples > num_source_points:
        indices = np.random.choice(num_source_points, size=num_samples, replace=True, p=probs)
    else:
        indices = np.random.choice(num_source_points, size=num_samples, replace=False, p=probs)
    
    # Get base samples
    source_sample = source_data[indices].clone().detach().to(dtype=torch.float32, device=device)
    
    # Add Gaussian noise for regularization
    sigma_matrix = sigma * torch.eye(dim, device=device)
    noise = torch.distributions.MultivariateNormal(
        loc=torch.zeros(dim, device=device),
        covariance_matrix=sigma_matrix
    ).sample((num_samples,))
    
    # Combine samples with noise
    source_sample = source_sample + noise

    
    return source_sample

def get_exact_batch(num_samples, device, data):
    """
    Get exact batch from data
    
    Args:
        num_samples: Number of samples to get
        device: Computation device
        data: Data to get batch from
    """
    indices = np.random.choice(len(data), size=num_samples, replace=True)
    z0_pca = data[indices].clone().detach().to(dtype=torch.float32, device=device)
    logp_diff_t0 = torch.zeros(num_samples, 1).to(device)
    return z0_pca, logp_diff_t0

def get_batch_by_index(indices, device, data):
    """
    Get a batch from data using fixed indices, with index wrap-around.

    Args:
        indices (array-like): List or array of sample indices to select.
        device (str or torch.device): Device to move the tensors to.
        data (Tensor): Full dataset (2D tensor) to sample from.

    Returns:
        z0_pca (Tensor): Selected data batch of shape [len(indices), ...]
        logp_diff_t0 (Tensor): Zero tensor with shape [len(indices), 1]
    """
    indices = torch.tensor(indices, dtype=torch.long)
    indices = indices % data.shape[0]  # wrap indices to avoid out-of-bounds
    z0_pca = data[indices].clone().detach().to(dtype=torch.float32, device=device)
    logp_diff_t0 = torch.zeros(len(indices), 1, dtype=torch.float32, device=device)
    return z0_pca, logp_diff_t0


def torch_record():
    ma = torch.cuda.memory_allocated()
    mma = torch.cuda.max_memory_allocated()
    mr = torch.cuda.memory_reserved()
    mmr = torch.cuda.max_memory_reserved()
    print(f"ma:{ma / 2 ** 20} MB    mma:{mma / 2 ** 20} MB    mr:{mr / 2 ** 20} MB    mmr:{mmr / 2 ** 20} MB")

import torch
import math
from geomloss import SamplesLoss


def normalize_to_unit_cube_global(data, time_list, save_path='norm_params.pt', device='cpu', norm_params=None):
    """
    Normalize a list of tensors using W2-based global scale.

    Args:
        data: list of tensors, each of shape (n_samples, n_features)
        save_path: path to save computed norm_params (only used when norm_params is None)
        device: device for W2 computation
        norm_params: if provided, reuse this scale instead of computing a new one

    Returns:
        normalized_data (list), norm_params
    """

    if norm_params is None:
        # Compute scale from data
        w2_loss = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=0.0001,
            backend="tensorized"
        ).to(device)

        Sum_w2 = 0.0

        # compute total W2^2
        for i in range(len(data) - 1):
            x = data[i].to(device)
            y = data[i + 1].to(device)
            time_interval = time_list[i + 1] - time_list[i]
            loss_val = w2_loss(x, y)
            Sum_w2 += loss_val.item()/time_interval

        scale = math.sqrt(Sum_w2 + 1e-12)
        norm_params = {"scale": scale}
        torch.save(norm_params, save_path)

    # normalize
    scale = norm_params["scale"]
    normalized_data = [d / scale for d in data]
    

    return normalized_data, norm_params

def denormalize_from_unit_cube_global(normalized_data, norm_params=None):
    """
    Recover original list of tensors
    """

    if norm_params is None:
        norm_params = torch.load('norm_params.pt')

    scale = norm_params["scale"]

    data = [d * scale for d in normalized_data]

    return data