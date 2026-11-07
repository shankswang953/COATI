"""Fresh Gaussian observations and AnnData assembly for the comparison example."""
import anndata as ad
import numpy as np
import pandas as pd
import coati as ct
from generate_gaussian_data import make_data, map_to_secondary, make_support


def assemble_anndata(time0, time1, support2d, support3d):
    primary = np.concatenate([time0, time1])
    secondary = np.concatenate([map_to_secondary(x, force_z_zero=True) for x in [time0, time1]])
    counts = [len(time0), len(time1)]
    identifiers = pd.Index([f"toy-{i:05d}" for i in range(len(primary))])
    labels = np.repeat(["Initial", "Terminal A", "Terminal B"],
                       [counts[0], counts[1] // 2, counts[1] - counts[1] // 2])
    cells = ad.AnnData(primary.copy(), obs=pd.DataFrame({
        "time": np.repeat([0., 1.], counts), "celltype": pd.Categorical(labels),
        "barcode": identifiers.to_numpy()}, index=identifiers))
    cells.obsm["X_latent"] = primary.copy()
    cells = ct.prepare_anndata(cells, use_rep="X_latent", secondary=secondary,
                                secondary_key="X_secondary", warn=False)
    map_ids = pd.Index([*identifiers, *[f"support-{i:05d}" for i in range(len(support2d))]])
    pairs = ad.AnnData(np.concatenate([primary, support2d]), obs=pd.DataFrame({
        "source": pd.Categorical(np.repeat(["Observed", "Support"], [len(primary), len(support2d)])),
        "barcode": map_ids.to_numpy()}, index=map_ids))
    pairs.obsm["X_secondary"] = np.concatenate([secondary, support3d])
    return cells, pairs


def generate_fresh_anndata():
    time0, time1 = make_data(seed=2026, n_initial=1000)
    support2d, support3d = make_support(np.random.default_rng(2026), 4000, 3000, .13, .08, 1.)
    return assemble_anndata(time0, time1, support2d, support3d)
