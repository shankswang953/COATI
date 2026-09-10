import torch
import numpy as np

def normalize_temporal_data(data_list, normalization_type='global', target_variance=1.0, 
                             time_labels=None, verbose=True):
    """
    Normalize the temporal data, support variance control.
    
    Parameters:
        data_list: list of tensors, each tensor contains the data of a time point
        normalization_type: normalization type
            'global': use the global statistics of all time points to normalize
            'per_timepoint': normalize each time point independently
        target_variance: target variance, for controlling the variance of the output data
        time_labels: list of time point labels, for output information
        verbose: whether to print detailed information
    
    Returns:
        normalized_data: list of normalized data
    """
    
    # convert to numpy for processing
    data_numpy = []
    for item in data_list:
        if isinstance(item, torch.Tensor):
            data_numpy.append(item.detach().cpu().numpy())
        else:
            data_numpy.append(item)
    
    if time_labels is None:
        time_labels = [f'time_{i}' for i in range(len(data_list))]
    
    # create the list of normalized data
    normalized_data = []
    
    if normalization_type == 'global':
        # concatenate all time points data
        all_data = np.concatenate(data_numpy, axis=0)
        
        # calculate the global statistics
        global_mean = all_data.mean(axis=0)
        global_std = all_data.std(axis=0)
        
        # avoid division by zero error
        global_std = np.where(global_std < 1e-10, 1.0, global_std)
        
        # normalize each time point
        for i, data in enumerate(data_numpy):
            normalized = (data - global_mean) / global_std
            
            # control the variance
            if target_variance != 1.0:
                current_var = normalized.var()
                scale_factor = np.sqrt(target_variance / current_var)
                normalized = normalized * scale_factor
            
            normalized_data.append(normalized)
            
            if verbose:
                print(f"Time point {time_labels[i]} normalized - mean: {normalized.mean():.4f}, variance: {normalized.var():.4f}")
    
    elif normalization_type == 'per_timepoint':
        # normalize each time point independently
        for i, data in enumerate(data_numpy):
            local_mean = data.mean(axis=0)
            local_std = data.std(axis=0)
            
            # avoid division by zero error
            local_std = np.where(local_std < 1e-10, 1.0, local_std)
            
            normalized = (data - local_mean) / local_std
            
            # control the variance
            if target_variance != 1.0:
                current_var = normalized.var()
                scale_factor = np.sqrt(target_variance / current_var)
                normalized = normalized * scale_factor
            
            normalized_data.append(normalized)
            
            if verbose:
                print(f"Time point {time_labels[i]} normalized - mean: {normalized.mean():.4f}, variance: {normalized.var():.4f}")
    
    
    # convert back to the original data type
    for i, (item, normalized) in enumerate(zip(data_list, normalized_data)):
        if isinstance(item, torch.Tensor):
            normalized_data[i] = torch.tensor(normalized, dtype=item.dtype, device=item.device)
    
    return normalized_data