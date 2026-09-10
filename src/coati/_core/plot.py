

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np




def visualize_evolution(all_samples, terminal_samples=None, save_path='density_evolution.gif', 
                      viz_timesteps=21, fps=10, dpi=100):
    """
    Visualize the evolution of samples over time using animation
    
    Args:
        all_samples (torch.Tensor): Tensor of shape (total_timesteps, num_samples, dim)
        terminal_samples (torch.Tensor, optional): Terminal samples to plot as reference
        save_path (str): Path to save the animation
        viz_timesteps (int): Number of timesteps to visualize
        fps (int): Frames per second for the animation
        dpi (int): Dots per inch for the animation
    """
    # Convert to numpy for visualization
    min_val = all_samples.min()
    max_val = all_samples.max()
    min_val_terminal = terminal_samples.min()
    max_val_terminal = terminal_samples.max()
    xmin = min(min_val, min_val_terminal) - 0.05*abs(min(min_val, min_val_terminal))
    xmax = max(max_val, max_val_terminal) + 0.05*abs(max(max_val, max_val_terminal))
    ymin = min(min_val, min_val_terminal) - 0.05*abs(min(min_val, min_val_terminal))
    ymax = max(max_val, max_val_terminal) + 0.05*abs(max(max_val, max_val_terminal))
    

    if terminal_samples is not None:
        terminal_samples = terminal_samples.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot initial and terminal samples if provided
    if terminal_samples is not None:
        ax.scatter(terminal_samples[:, 0], terminal_samples[:, 1], 
                  color='blue', alpha=0.1, label='Terminal', s=3)
    ax.scatter(all_samples[0, :, 0], all_samples[0, :, 1], 
              color='red', alpha=0.1, label='Initial', s=3)
    
    # Set plot properties
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title('Sample Evolution')
    ax.set_xlabel('X-axis')
    ax.set_aspect('equal')
    ax.grid(False)
    ax.legend()
    
    # Create scatter plots for animation - one for initial points and one for terminal points
    scat_initial = ax.scatter([], [], color='red', alpha=0.7, label='Moving Points')
    if terminal_samples is not None:
        scat_terminal = ax.scatter([], [], color='blue', alpha=0.7, label='Terminal Points')
    
    def init():
        scat_initial.set_offsets(np.empty((0, 2)))
        if terminal_samples is not None:
            scat_terminal.set_offsets(np.empty((0, 2)))
        return (scat_initial,) if terminal_samples is None else (scat_initial, scat_terminal)
    
    def update(frame):
        # Update initial points
        scat_initial.set_offsets(all_samples[frame, :, :])
        scat_initial.set_sizes([0.1] * all_samples.shape[1])
        
        # Update terminal points if provided
        if terminal_samples is not None:
            scat_terminal.set_offsets(terminal_samples)
            scat_terminal.set_sizes([0.1] * terminal_samples.shape[0])
        
        return (scat_initial,) if terminal_samples is None else (scat_initial, scat_terminal)
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=all_samples.shape[0],
                                init_func=init, blit=False, interval=100)
    
    # Save animation with optimized settings
    writer = animation.PillowWriter(fps=fps)
    ani.save(save_path, writer=writer, dpi=dpi)
    plt.close(fig)
    
    return HTML(ani.to_jshtml())

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_and_distribution(
    data_train, 
    all_samples, 
    plot_space=[0, 1], 
    n_trajectories=None,
    title=None,            
    xlabel='Dimension 1',        
    ylabel='Dimension 2',        
    linewidth=0.5,
    label_fontsize=10,           
    axis_fontsize=12,
    tick_fontsize=10,
    grid=True,
    x_plot=None,                
    y_plot=None,
    legend_loc='best',
    data_size=30,
    data_transparency=0.3,
    colors=None,
    save_path=None,
    # --- new params ---
    time_indices=None,     # e.g. [0, 1, 5]
    time_labels=None       # e.g. ['t0', 't1', 't5'] or custom labels
):
    """
    Plot selected distributions and trajectories with customizable display range, time selection, and labels.

    Args:
        data_train: list of arrays, each of shape (n_samples, n_dims)
        all_samples: numpy array of shape (n_timepoints, n_trajectories, n_dims)
        time_indices: list of integers specifying which time indices in data_train to plot
                      (default: all)
        time_labels: list of strings specifying labels for each plotted time index
                     (default: auto-generated as t=0, t=1, ...)
        plot_space: dimensions to plot (default: [0, 1])
        n_trajectories: Optional parameter to limit number of trajectories (default: all)
    """
    plt.figure(figsize=(10, 10))
    dim1, dim2 = plot_space

    n_timepoints = len(data_train)

    # --- handle time selection ---
    if time_indices is None:
        time_indices = list(range(n_timepoints))
    if time_labels is None:
        time_labels = [f't={i}' for i in time_indices]
    else:
        if len(time_labels) != len(time_indices):
            raise ValueError("Length of time_labels must match length of time_indices.")

    # --- colors ---
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0., 1., len(time_indices)))

    # --- Plot selected distributions ---
    for i, idx in enumerate(time_indices):
        data = data_train[idx]
        plt.scatter(
            data[:, dim1], data[:, dim2],
            alpha=data_transparency, color=colors[i],
            label=time_labels[i], s=data_size
        )

    # --- Select trajectories ---
    if n_trajectories is None or n_trajectories > all_samples.shape[1]:
        n_trajectories = all_samples.shape[1]
    selected_indices = np.random.choice(all_samples.shape[1], n_trajectories, replace=False)

    # --- Plot trajectories ---
    for idx in selected_indices:
        trajectory = all_samples[:, idx, :]
        plt.plot(
            trajectory[:, dim1], trajectory[:, dim2],
            color='black', alpha=0.5, linewidth=linewidth
        )
        plt.scatter(trajectory[1:-1, dim1], trajectory[1:-1, dim2],
                    color='black', s=data_size, alpha=0.2, marker='.')
        plt.scatter(trajectory[0, dim1], trajectory[0, dim2],
                    color='black', s=data_size * 3, alpha=1, marker='*')
        plt.scatter(trajectory[-1, dim1], trajectory[-1, dim2],
                    color='black', s=data_size * 3, alpha=1, marker='^')

    # --- Labels, title, legend ---
    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)
    if title:
        plt.title(title, fontsize=axis_fontsize + 5)

    legend = plt.legend(fontsize=label_fontsize, markerscale=3, loc=legend_loc)
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    if grid:
        plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # --- Axis limits ---
    if x_plot is not None:
        plt.xlim(x_plot[0], x_plot[1])
    else:
        all_x = np.concatenate([data_train[i][:, dim1] for i in time_indices] +
                               [all_samples[:, :, dim1].flatten()])
        margin_x = 0.1
        plt.xlim(all_x.min() - margin_x, all_x.max() + margin_x)

    if y_plot is not None:
        plt.ylim(y_plot[0], y_plot[1])
    else:
        all_y = np.concatenate([data_train[i][:, dim2] for i in time_indices] +
                               [all_samples[:, :, dim2].flatten()])
        margin_y = 0.1
        plt.ylim(all_y.min() - margin_y, all_y.max() + margin_y)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_temporal_embeddings(
    data_list, 
    time_labels=None, 
    colors=None, 
    figsize=(15, 20), 
    alpha=0.5, 
    title="Temporal Embedding Visualization", 
    marker='o', 
    s=30, 
    TwoDplot=True
):
    """
    Plot the temporal embeddings of the data in 2D or 3D.

    Parameters:
        data_list: list of tensors or arrays, each contains the data of a time point
        time_labels: optional, list of time point labels, default is 't0, t1, t2...'
        colors: optional, list of colors, default is the color cycle of matplotlib
        figsize: figure size, default (15, 20)
        alpha: point opacity, default 0.5
        title: figure title, default "Temporal Embedding Visualization"
        marker: scatter plot marker, default 'o'
        s: point size, default 30
        TwoDplot: if True, plot 2D; if False, plot 3D
    Returns:
        fig, ax: matplotlib figure and axis object, can be further customized
    """
    # Check the dimension of the data
    for i, data in enumerate(data_list):
        if TwoDplot:
            if data.shape[1] < 2:
                print(f" Warning: The dataset {i} is not two-dimensional. Shape: {data.shape}")
        else:
            if data.shape[1] < 3:
                print(f" Warning: The dataset {i} is not three-dimensional. Shape: {data.shape}")
    # Select the first 2 or 3 dimensions for plotting
    if TwoDplot:
        data_plot = [data[:, :2] for data in data_list]
    else:
        data_plot = [data[:, :3] for data in data_list]

    # Create default labels if not provided
    if time_labels is None:
        time_labels = [f't{i}' for i in range(len(data_list))]

    # Create default colors list if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        if len(colors) < len(data_list):
            colors = list(mcolors.TABLEAU_COLORS.values()) * ((len(data_list) // len(mcolors.TABLEAU_COLORS)) + 1)
            colors = colors[:len(data_list)]

    # Create the figure and axis
    if TwoDplot:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Plot the data for each time point
    for i, data in enumerate(data_plot):
        color_idx = i % len(colors)  # Cycle through colors if not enough
        data_np = data.detach().cpu().numpy() if hasattr(data, 'detach') else data
        if TwoDplot:
            ax.scatter(data_np[:, 0], data_np[:, 1], 
                       color=colors[color_idx], 
                       alpha=alpha, 
                       label=time_labels[i],
                       marker=marker,
                       s=s)
        else:
            ax.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2],
                       color=colors[color_idx], 
                       alpha=alpha, 
                       label=time_labels[i],
                       marker=marker,
                       s=s)

    # Add legend and labels
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    if not TwoDplot:
        ax.set_zlabel("Dimension 3", fontsize=12)
        ax.set_box_aspect([1,1,1])  # Keep xyz aspect ratio equal
        # Optionally, you can set the view angle:
        # ax.view_init(elev=20, azim=120)
    else:
        ax.set_aspect('equal')

    return fig, ax

def plot_embeddings(
    data_list, 
    time_labels=None, 
    colors=None, 
    figsize=(15, 20), 
    alpha=0.5, 
    title="Temporal Embedding Visualization", 
    marker='o', 
    s=30, 
    TwoDplot=True
):
    """
    Plot the temporal embeddings of the data in 2D or 3D.

    Parameters:
        data_list: list of tensors or arrays, each contains the data of a time point
        time_labels: optional, list of time point labels, default is 't0, t1, t2...'
        colors: optional, list of colors, default is the color cycle of matplotlib
        figsize: figure size, default (15, 20)
        alpha: point opacity, default 0.5
        title: figure title, default "Temporal Embedding Visualization"
        marker: scatter plot marker, default 'o'
        s: point size, default 30
        TwoDplot: if True, plot 2D; if False, plot 3D
    Returns:
        fig, ax: matplotlib figure and axis object, can be further customized
    """
    # Check the dimension of the data
    for i, data in enumerate(data_list):
        if TwoDplot:
            if data.shape[1] < 2:
                print(f" Warning: The dataset {i} is not two-dimensional. Shape: {data.shape}")
        else:
            if data.shape[1] < 3:
                print(f" Warning: The dataset {i} is not three-dimensional. Shape: {data.shape}")
    # Select the first 2 or 3 dimensions for plotting
    if TwoDplot:
        data_plot = [data[:, [1, 2]] for data in data_list]
    else:
        data_plot = [data[:, :3] for data in data_list]

    # Create default labels if not provided
    if time_labels is None:
        time_labels = [f't{i}' for i in range(len(data_list))]

    # Create default colors list if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        if len(colors) < len(data_list):
            colors = list(mcolors.TABLEAU_COLORS.values()) * ((len(data_list) // len(mcolors.TABLEAU_COLORS)) + 1)
            colors = colors[:len(data_list)]

    # Create the figure and axis
    if TwoDplot:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Plot the data for each time point
    for i, data in enumerate(data_plot):
        color_idx = i % len(colors)  # Cycle through colors if not enough
        data_np = data.detach().cpu().numpy() if hasattr(data, 'detach') else data
        if TwoDplot:
            ax.scatter(data_np[:, 0], data_np[:, 1],
                       color=colors[color_idx],
                       alpha=alpha,
                       label=time_labels[i],
                       marker=marker,
                       s=s)
        else:
            ax.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2],
                       color=colors[color_idx],
                       alpha=alpha,
                       label=time_labels[i],
                       marker=marker,
                       s=s)

    # Add legend and labels
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    if not TwoDplot:
        ax.set_zlabel("Dimension 3", fontsize=12)
        ax.set_box_aspect([1,1,1])  # Keep xyz aspect ratio equal
        # Optionally, you can set the view angle:
        # ax.view_init(elev=20, azim=120)
    else:
        ax.set_aspect('equal')

    return fig, ax


def plot_umap_trajectories(
    all_samples_forward,
    adata,
    umap_model,
    time_points,
    pred_time_points=None,
    norm_params=None,
    denormalize=True,
    celltype_key='celltype_sub',
    stage_key='stage',
    n_trajectories=None,
    title=None,
    figsize=(20, 10),
    dpi=100,
    xlim=None,
    ylim=None,
    bg_alpha=0.3,
    bg_size=5,
    traj_alpha=0.7,
    traj_linewidth=1,
    pred_size=50,
    pred_alpha=0.9,
    save_path=None,
):
    """
    Plot two side-by-side UMAP panels with trajectories and predicted points:
      Left:  background colored by celltype_key (tab20 colormap).
      Right: background colored by stage_key (plasma / blue-yellow gradient).

    Both panels overlay ODE trajectory lines and predicted distributions at
    the requested real time points.

    Args:
        all_samples_forward: tensor/array (total_steps, n_cells, D).
        adata: AnnData with .obsm['X_umap'], obs[celltype_key], obs[stage_key].
        umap_model: fitted reducer with .transform().
        time_points: list of ALL real time points covering the trajectory,
            e.g. [0, 1, 1.5, 2].  Used to compute dt.
        pred_time_points: list of real time points at which to mark predictions.
            Defaults to time_points (i.e. mark every real time point).
        norm_params: for denormalize_from_unit_cube_global (required if denormalize).
        denormalize: whether to denormalize before UMAP.
        celltype_key: adata.obs column for cell-type coloring (left panel).
        stage_key: adata.obs column for stage/time coloring (right panel).
        n_trajectories: how many trajectories to draw (default: all).
        title: suptitle for the whole figure.
        figsize: figure size for the two-panel figure.
        dpi: figure DPI.
        xlim / ylim: axis limits (applied to both panels).
        bg_alpha / bg_size: background scatter style.
        traj_alpha / traj_linewidth: trajectory line style.
        pred_size / pred_alpha: predicted-point marker style.
        save_path: if provided, save figure.

    Returns:
        fig, (ax1, ax2), umap_trajectory
    """
    import torch
    from .DataLoad import denormalize_from_unit_cube_global

    if pred_time_points is None:
        pred_time_points = list(time_points)

    if hasattr(all_samples_forward, 'detach'):
        all_samples_np = all_samples_forward.detach().cpu().numpy()
    else:
        all_samples_np = np.array(all_samples_forward)

    total_steps, n_cells, D = all_samples_np.shape

    # --- UMAP transform all time steps ---
    umap_trajectory = np.zeros((total_steps, n_cells, 2))
    for t in range(total_steps):
        sample_slice = all_samples_np[t]
        if denormalize:
            if norm_params is None:
                raise ValueError("norm_params required when denormalize=True")
            sample_slice = denormalize_from_unit_cube_global(
                torch.tensor(sample_slice, dtype=torch.float32), norm_params
            )
            if hasattr(sample_slice, 'numpy'):
                sample_slice = sample_slice.numpy()
        umap_trajectory[t] = umap_model.transform(sample_slice)

    # --- Compute trajectory indices for each prediction time point ---
    t_start, t_end = time_points[0], time_points[-1]
    dt = (t_end - t_start) / (total_steps - 1)
    pred_indices = []
    for tp in pred_time_points:
        idx = int(round((tp - t_start) / dt))
        idx = min(idx, total_steps - 1)
        pred_indices.append(idx)

    # --- Select trajectories ---
    if n_trajectories is None or n_trajectories > n_cells:
        n_trajectories = n_cells
    sel = np.random.choice(n_cells, n_trajectories, replace=False)

    # --- Prepare background data ---
    umap_coords = adata.obsm['X_umap']
    celltypes = adata.obs[celltype_key].values
    stages = adata.obs[stage_key].values

    unique_ct = np.unique(celltypes)
    ct_cmap = plt.cm.tab20
    ct_colors = {ct: ct_cmap(i / max(len(unique_ct) - 1, 1))
                 for i, ct in enumerate(unique_ct)}

    unique_stages = np.sort(np.unique(stages))
    stage_cmap = plt.cm.viridis
    stage_colors = {s: stage_cmap(i / max(len(unique_stages) - 1, 1))
                    for i, s in enumerate(unique_stages)}

    # --- Helper: draw trajectories + predictions on a given axes ---
    def _draw_overlay(ax):
        # Trajectory lines
        for idx in sel:
            traj = umap_trajectory[:, idx, :]
            ax.plot(traj[:, 0], traj[:, 1],
                    color='black', linewidth=traj_linewidth, alpha=traj_alpha)
        # Predicted points at each requested time
        pred_cm = plt.cm.cool
        for i, (tp, ri) in enumerate(zip(pred_time_points, pred_indices)):
            color = pred_cm(i / max(len(pred_time_points) - 1, 1))
            pts = umap_trajectory[ri, sel, :]
            ax.scatter(pts[:, 0], pts[:, 1],
                       s=pred_size, color=color, alpha=pred_alpha,
                       edgecolors='black', linewidths=0.5,
                       label=f'Pred t={tp}', zorder=5)

    # ======================== Two-panel figure ========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # --- Left panel: colored by cell type ---
    for ct in unique_ct:
        mask = celltypes == ct
        ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                    s=bg_size, alpha=bg_alpha, color=ct_colors[ct],
                    marker='x', label=ct, rasterized=True)
    _draw_overlay(ax1)
    ax1.set_title(f'Colored by {celltype_key}', fontsize=14)

    # --- Right panel: colored by stage (blue-yellow gradient) ---
    for s in unique_stages:
        mask = stages == s
        ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                    s=bg_size, alpha=bg_alpha, color=stage_colors[s],
                    marker='x', label=s, rasterized=True)
    _draw_overlay(ax2)
    ax2.set_title(f'Colored by {stage_key}', fontsize=14)

    # --- Shared formatting ---
    for ax in (ax1, ax2):
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        # Reorder legend: predictions first, then background labels
        handles, labels = ax.get_legend_handles_labels()
        pred_mask = [l.startswith('Pred') for l in labels]
        pred_h = [h for h, m in zip(handles, pred_mask) if m]
        pred_l = [l for l, m in zip(labels, pred_mask) if m]
        bg_h = [h for h, m in zip(handles, pred_mask) if not m]
        bg_l = [l for l, m in zip(labels, pred_mask) if not m]
        ax.legend(pred_h + bg_h, pred_l + bg_l,
                  fontsize=7, markerscale=2, loc='best',
                  ncol=2, framealpha=0.7)

    if title is not None:
        fig.suptitle(title, fontsize=18, y=1.02)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight')

    return fig, (ax1, ax2), umap_trajectory


def animate_umap_trajectories(
    all_samples_forward,
    adata,
    umap_model,
    time_points,
    norm_params=None,
    denormalize=True,
    celltype_key='celltype_sub',
    stage_key='stage',
    umap_trajectory=None,
    title=None,
    figsize=(20, 10),
    dpi=100,
    xlim=None,
    ylim=None,
    bg_alpha=0.2,
    bg_size=5,
    pred_size=30,
    pred_alpha=0.9,
    fps=5,
    save_path='umap_evolution.gif',
):
    """
    Animate the UMAP evolution of predicted cells across ODE time steps.
    Each frame shows the predicted cell distribution at one time step.
    Two panels: left colored by celltype, right colored by stage.

    Args:
        all_samples_forward: tensor/array (total_steps, n_cells, D).
        adata: AnnData with .obsm['X_umap'], obs[celltype_key], obs[stage_key].
        umap_model: fitted reducer with .transform().
        time_points: list of real time points, e.g. [0, 1, 1.5, 2].
        norm_params: for denormalization (required if denormalize=True).
        denormalize: whether to denormalize before UMAP.
        celltype_key: adata.obs column for cell-type coloring (left panel).
        stage_key: adata.obs column for stage coloring (right panel).
        umap_trajectory: precomputed (total_steps, n_cells, 2) array.
            If provided, skip UMAP transform to save time.
        title: base title for the figure.
        figsize / dpi: figure size and resolution.
        xlim / ylim: axis limits (auto-computed from data if None).
        bg_alpha / bg_size: background scatter style.
        pred_size / pred_alpha: predicted-point marker style.
        fps: frames per second for the animation.
        save_path: path to save gif/mp4. None to skip saving.

    Returns:
        anim: matplotlib FuncAnimation object
        umap_trajectory: (total_steps, n_cells, 2) array
    """
    import torch
    from .DataLoad import denormalize_from_unit_cube_global

    # --- UMAP transform (reuse if provided) ---
    if umap_trajectory is None:
        if hasattr(all_samples_forward, 'detach'):
            all_samples_np = all_samples_forward.detach().cpu().numpy()
        else:
            all_samples_np = np.array(all_samples_forward)

        total_steps, n_cells, D = all_samples_np.shape
        umap_trajectory = np.zeros((total_steps, n_cells, 2))
        for t in range(total_steps):
            sample_slice = all_samples_np[t]
            if denormalize:
                if norm_params is None:
                    raise ValueError("norm_params required when denormalize=True")
                sample_slice = denormalize_from_unit_cube_global(
                    torch.tensor(sample_slice, dtype=torch.float32), norm_params
                )
                if hasattr(sample_slice, 'numpy'):
                    sample_slice = sample_slice.numpy()
            umap_trajectory[t] = umap_model.transform(sample_slice)

    total_steps, n_cells, _ = umap_trajectory.shape

    # --- Time axis ---
    t_start, t_end = time_points[0], time_points[-1]
    frame_times = np.linspace(t_start, t_end, total_steps)

    # --- Background data ---
    umap_coords = adata.obsm['X_umap']
    celltypes = adata.obs[celltype_key].values
    stages = adata.obs[stage_key].values

    unique_ct = np.unique(celltypes)
    ct_cmap = plt.cm.tab20
    ct_colors = {ct: ct_cmap(i / max(len(unique_ct) - 1, 1))
                 for i, ct in enumerate(unique_ct)}

    unique_stages = np.sort(np.unique(stages))
    stage_cmap = plt.cm.viridis
    stage_colors = {s: stage_cmap(i / max(len(unique_stages) - 1, 1))
                    for i, s in enumerate(unique_stages)}

    # --- Auto axis limits ---
    if xlim is None:
        all_x = np.concatenate([umap_coords[:, 0], umap_trajectory[:, :, 0].ravel()])
        margin = (all_x.max() - all_x.min()) * 0.05
        xlim = (all_x.min() - margin, all_x.max() + margin)
    if ylim is None:
        all_y = np.concatenate([umap_coords[:, 1], umap_trajectory[:, :, 1].ravel()])
        margin = (all_y.max() - all_y.min()) * 0.05
        ylim = (all_y.min() - margin, all_y.max() + margin)

    # --- Build figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Draw static background once
    for ct in unique_ct:
        mask = celltypes == ct
        ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                    s=bg_size, alpha=bg_alpha, color=ct_colors[ct],
                    marker='x', rasterized=True)
    for s in unique_stages:
        mask = stages == s
        ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                    s=bg_size, alpha=bg_alpha, color=stage_colors[s],
                    marker='x', rasterized=True)

    # Scatter handles for predicted points (updated each frame)
    scat1 = ax1.scatter([], [], s=pred_size, alpha=pred_alpha,
                        color='red', edgecolors='black', linewidths=0.3, zorder=5)
    scat2 = ax2.scatter([], [], s=pred_size, alpha=pred_alpha,
                        color='red', edgecolors='black', linewidths=0.3, zorder=5)

    for ax in (ax1, ax2):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    ax1.set_title(f'Colored by {celltype_key}', fontsize=14)
    ax2.set_title(f'Colored by {stage_key}', fontsize=14)

    time_text = fig.suptitle('', fontsize=16)

    def _update(frame_idx):
        pts = umap_trajectory[frame_idx]
        scat1.set_offsets(pts)
        scat2.set_offsets(pts)
        t_label = f't = {frame_times[frame_idx]:.3f}'
        if title is not None:
            time_text.set_text(f'{title}  |  {t_label}')
        else:
            time_text.set_text(t_label)
        return scat1, scat2, time_text

    anim = animation.FuncAnimation(
        fig, _update, frames=total_steps, interval=1000 // fps, blit=False,
    )

    if save_path is not None:
        if save_path.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps)
        else:
            writer = animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        print(f"[animate_umap] saved -> {save_path}")

    return anim, umap_trajectory

import scanpy as sc
def plot_umap_highlight(adata, groupby, targets, size=10, ncols=3):
    n = len(targets)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.flatten()
    
    for i, target in enumerate(targets):
        colname = f"{groupby}_highlight_{target}"
        
        adata.obs[colname] = adata.obs[groupby].astype(str)
        adata.obs.loc[adata.obs[colname] != target, colname] = "Other"
        adata.obs[colname] = adata.obs[colname].astype("category")
        
        palette = {
            target: "#d62728",
            "Other": "#d3d3d3"
        }
        
        sc.pl.umap(
            adata,
            color=colname,
            palette=palette,
            size=size,
            frameon=False,
            ax=axes[i],
            show=False,
            title=target
        )
    
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()