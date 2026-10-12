"""UMAP views of learned motion, instantaneous growth and supplied cell types."""
import numpy as np


def _cells(adata, *, use_rep, time_key, time, time_mapping, basis, celltype_key):
    from .anndata_io import resolve_representation, observation_times
    from .velocity import _matrix
    rep = resolve_representation(adata, use_rep)
    points = _matrix(adata.X if rep is None else adata.obsm[rep], "training representation")
    key = basis if basis.startswith("X_") else "X_" + basis
    if key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{key!r}]; prepare_anndata(compute_umap=True) can fit a display UMAP.")
    coordinates = _matrix(adata.obsm[key], "embedding", 2)
    if len(coordinates) != adata.n_obs or len(points) != adata.n_obs:
        raise ValueError("Representation and embedding rows must match AnnData observations.")
    if celltype_key not in adata.obs:
        raise KeyError(f"Missing adata.obs[{celltype_key!r}]; use prepare_anndata to standardize labels.")
    import pandas as pd
    celltypes = pd.Categorical(adata.obs[celltype_key])
    if np.any(celltypes.codes < 0):
        raise ValueError("Cell types cannot be missing; use 'unknown' for unlabeled cells.")
    times = observation_times(adata, time=time, time_key=time_key, time_mapping=time_mapping)
    return points, coordinates, times, celltypes, rep


def _growth_panel(result, points, coordinates, times, celltypes, *, field=None,
                  basis="umap", labels=True, ax=None, **options):
    from matplotlib.colors import Normalize
    from .velocity import VelocityEmbedding, annotate_celltypes
    growth = result.compute_growth(points, time=times)
    bound = max(float(np.abs(growth).max()), 1e-12)
    if field is None:
        if options.get("style", "cells") != "cells":
            raise ValueError("Set show_velocity=True to draw learned streams or arrows on growth.")
        field = VelocityEmbedding(coordinates, np.zeros_like(coordinates), times, basis.removeprefix("X_"))
        options.setdefault("style", "cells")
    else:
        options.setdefault("style", "stream")
    options.setdefault("cmap", "coolwarm")
    options.setdefault("norm", Normalize(-bound, bound))
    options.setdefault("colorbar_label", "Growth rate · per time unit")
    options.setdefault("title", "Instantaneous growth")
    ax = field.plot(color=growth, ax=ax, **options)
    if labels:
        annotate_celltypes(ax, coordinates, celltypes)
    return ax


def plot_growth(result, adata, *, use_rep="auto", time_key="time", time=None,
                time_mapping=None, basis="umap", celltype_key="celltype", labels=True,
                show_velocity=False, n_neighbors=30, ridge=1e-3, ax=None, **plot_options):
    """Plot per-cell d log(weight)/dt, with optional COATI streams and labels.

    The heat map evaluates growth on the cells' original model features, never
    on UMAP coordinates. Positive values increase mass; negative values reduce
    mass. The colors are instantaneous rates, not accumulated trajectory weights.
    By default each cell uses its own observation time. Pass time for a common
    time view, or subset AnnData to one stage. AnnData is never modified.
    """
    points, coordinates, times, celltypes, rep = _cells(adata, use_rep=use_rep,
        time_key=time_key, time=time, time_mapping=time_mapping, basis=basis, celltype_key=celltype_key)
    field = result.compute_velocity_embedding(adata, use_rep=rep, time=times, basis=basis,
        n_neighbors=n_neighbors, ridge=ridge) if show_velocity else None
    return _growth_panel(result, points, coordinates, times, celltypes, field=field, basis=basis,
                         labels=labels, ax=ax, n_neighbors=n_neighbors, **plot_options)


def plot_dynamics(result, adata, *, use_rep="auto", time_key="time", time=None,
                  time_mapping=None, basis="umap", celltype_key="celltype", labels=True,
                  include_growth=None, palette=None, n_neighbors=30, ridge=1e-3,
                  axes=None, **plot_options):
    """Show cell-type-colored COATI streams beside a growth heat map with streams.

    Unbalanced models produce two panels; balanced models produce the motion
    panel only. Both panels share coordinates and limits. Returns a flat array
    of Matplotlib Axes. include_growth=True requires an unbalanced model.
    The motion is a local projection of COATI velocities, not scVelo's RNA
    splicing model or transition-probability computation. No AnnData is mutated.
    """
    import matplotlib.pyplot as plt
    points, coordinates, times, celltypes, rep = _cells(adata, use_rep=use_rep,
        time_key=time_key, time=time, time_mapping=time_mapping, basis=basis, celltype_key=celltype_key)
    unbalanced = bool(result.metadata["config"]["unbalancedModel"])
    if include_growth is None:
        include_growth = unbalanced
    if not isinstance(include_growth, bool):
        raise TypeError("include_growth must be a boolean or None.")
    if include_growth and not unbalanced:
        raise ValueError("Growth visualization requires balanced=False during training.")
    panels = 2 if include_growth else 1
    if axes is None:
        _, axes = plt.subplots(1, panels, figsize=(7 * panels, 5.5), layout="constrained", squeeze=False)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    if len(axes) != panels:
        raise ValueError(f"Provide {panels} axes for this model/view.")
    if len({id(ax.figure) for ax in axes}) != 1:
        raise ValueError("All dynamics axes must belong to the same figure.")
    field = result.compute_velocity_embedding(adata, use_rep=rep, time=times, basis=basis,
                                              n_neighbors=n_neighbors, ridge=ridge)
    if palette is None:
        palette = adata.uns.get(f"{celltype_key}_colors")
    options = dict(plot_options)
    options.setdefault("style", "stream")
    options.setdefault("point_size", 12)
    options.setdefault("alpha", .8)
    motion_options = dict(options)
    motion_options.setdefault("title", "Cell types · COATI flow")
    motion_options.setdefault("legend", not labels)
    field.plot(color=celltypes, palette=palette, labels=labels,
               ax=axes[0], n_neighbors=n_neighbors, **motion_options)
    if include_growth:
        _growth_panel(result, points, coordinates, times, celltypes, field=field,
            basis=basis, labels=labels, ax=axes[1], n_neighbors=n_neighbors, **options)
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    pad = np.maximum(.05 * (high - low), 1e-6)
    for ax in axes:
        ax.set(xlim=(low[0] - pad[0], high[0] + pad[0]), ylim=(low[1] - pad[1], high[1] + pad[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    return axes
