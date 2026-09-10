import numpy as np
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def compute_gene_metrics(
    gene_list,
    adata,
    adata_filter,
    nn_indices,
    stage_map,
    time_points,    
    T,
    S,
    time_min=0,
    time_max=2,
):

    evolution_time = np.linspace(time_min, time_max, T)
    stage_list = list(stage_map.keys())
    stage_time = np.array(time_points)
    stage_indices = np.array([
    np.argmin(np.abs(evolution_time - tp))
    for tp in stage_time
])

    results = []

    for gene_name in gene_list:

        if gene_name not in adata.raw.var_names:
            print(f"{gene_name} not found.")
            continue

        gene_idx = adata.raw.var_names.get_loc(gene_name)
        gene_col = adata.raw.X[:, gene_idx]

        # ===== trajectory reconstruction =====
        flat_nn_indices = nn_indices.reshape(-1)
        traj_expr = gene_col[flat_nn_indices].toarray().squeeze(1)
        traj_expr = traj_expr.reshape(T, S)

        traj_mean = traj_expr.mean(axis=1)

        # ===== variance =====
        total_variance = traj_expr.var(ddof=1)
        time_variance = traj_expr.var(axis=1, ddof=1)
        mean_time_variance = time_variance.mean()

        # ===== ground truth =====
        real_means = []

        for stage in stage_list:
            stage_mask = adata_filter.obs["stage"] == stage
            stage_expr = adata_filter.raw[stage_mask, gene_idx].X.toarray()
            real_means.append(stage_expr.mean())

        real_means = np.array(real_means)

        traj_at_stages = traj_mean[stage_indices]

        mse = np.mean((traj_at_stages - real_means)**2)
        rmse = np.sqrt(mse)

        results.append({
            "gene": gene_name,
            "RMSE": rmse,
            "TotalVariance": total_variance,
            "MeanTimeVariance": mean_time_variance
        })

    results_df = pd.DataFrame(results)

    print(results_df)

    return results_df


def sample_cell_indices(
    adata,
    cell_type,
    time=None,
    time_key=None,
    cell_type_key=None,
    n_cells=None,
    index_from_whole=False,
    filter_main=False,
    embedding_key="X_umap",
    outlier_fraction=0.02
):
    """
    Sample cell indices for a given time and cell type, 
    with optional filtering to keep only the main cluster in UMAP space.

    Args:
        adata: AnnData object
        cell_type: target cell type (string)
        time: target time (string)
        time_key: obs column name for time (default: "stage")
        cell_type_key: obs column name for cell type (default: "celltype")
        n_cells: number of cells to sample
        index_from_whole: if True, time filtering is skipped
        filter_main: if True, remove top outlier_fraction cells farthest from the cluster center
        embedding_key: key in adata.obsm to use as embedding space (default: "X_rna.umap")
        outlier_fraction: fraction of farthest cells to remove before sampling

    Returns:
        numpy array of sampled indices (relative to full adata)
    """

    # Default keys
    if time_key is None:
        time_key = "stage"
    if cell_type_key is None:
        cell_type_key = "celltype"

    # -----------------------------
    # Step 1: filter by time + cell type
    # -----------------------------
    if index_from_whole:
        adata_view = adata
        mask = (adata.obs[time_key] == time) & (adata.obs[cell_type_key] == cell_type)
        base_indices = np.where(mask)[0]
    else:
        adata_view = adata[adata.obs[time_key] == time]
        mask = (adata_view.obs[cell_type_key] == cell_type)
        base_indices = np.where(mask)[0]


    if len(base_indices) == 0:
        print(f"No cells found for time '{time}' and cell type '{cell_type}'.")
        return np.array([])

    # -----------------------------
    # Step 2: filter main cluster in UMAP
    # -----------------------------
    if filter_main:
        if embedding_key not in adata.obsm:
            raise ValueError(f"embedding_key '{embedding_key}' not found in adata.obsm.")

        emb = adata_view.obsm[embedding_key]              # shape (n_cells, dim)
        emb_sub = emb[base_indices, :]               # get this celltype's embedding

        # cluster center
        center = emb_sub.mean(axis=0)

        # euclidean distances to center
        dists = np.linalg.norm(emb_sub - center, axis=1)

        # number of cells to keep (remove farthest 2%)
        n_total = len(base_indices)
        n_keep = max(1, int((1 - outlier_fraction) * n_total))

        # pick the closest n_keep cells
        keep_order = np.argsort(dists)[:n_keep]
        base_indices = base_indices[keep_order]

    # -----------------------------
    # Step 3: random sampling
    # -----------------------------
    if len(base_indices) == 0:
        print("All cells removed during main-cluster filtering.")
        return np.array([])

    if n_cells is None:
        n_cells = len(base_indices)

    n_cells = min(n_cells, len(base_indices))
    sampled_indices = base_indices[:n_cells]

    return sampled_indices