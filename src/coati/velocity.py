"""Visualize learned primary-space velocities on existing 2D embeddings.

Projection uses a regularized local linear map estimated from corresponding
representation/embedding neighbors. It is a visualization approximation, not
an analytic UMAP Jacobian or an RNA splicing-kinetics estimate.
"""
from dataclasses import dataclass
import numpy as np


def _matrix(value, name, columns=None):
    if hasattr(value, "toarray"):
        value = value.toarray()
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or len(value) < 1 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite nonempty cells × features matrix.")
    if columns is not None and value.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns.")
    return value


@dataclass
class VelocityEmbedding:
    """Corresponding 2D positions and projected velocities, in unchanged row order."""
    coordinates: np.ndarray
    velocity: np.ndarray
    time: np.ndarray | None = None
    basis: str = "umap"
    projection: str = "local_linear"

    def plot(self, **kwargs):
        """Return a Matplotlib Axes using cells, grid arrows or streamlines."""
        return plot_velocity_embedding(self, **kwargs)


def project_velocity(points, velocity, embedding, *, n_neighbors=30, ridge=1e-3,
                     time=None, basis="umap"):
    """Project a vector using local weighted least-squares coordinate maps.

    All three arrays must contain the same cells in the same order. Local maps
    use neighbors in the model's input representation, without refitting UMAP.
    ridge is relative to the local displacement covariance; ridge=0 uses a
    minimum-norm solution. A degenerate input neighborhood produces zero flow.
    Velocities are not normalized, so a zero learned field remains zero.
    """
    from sklearn.neighbors import NearestNeighbors
    x = _matrix(points, "points")
    v = _matrix(velocity, "velocity", x.shape[1])
    y = _matrix(embedding, "embedding", 2)
    if len(x) != len(v) or len(x) != len(y) or len(x) < 3:
        raise ValueError("points, velocity and embedding need at least three corresponding rows.")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 2:
        raise ValueError("n_neighbors must be an integer of at least two.")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative.")
    neighbors = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(x))).fit(x)
    distance, indices = neighbors.kneighbors(x)
    projected = np.zeros((len(x), 2))
    for i, (dist, rows) in enumerate(zip(distance, indices)):
        keep = rows != i
        rows, dist = rows[keep][:n_neighbors], dist[keep][:n_neighbors]
        dx, dy = x[rows] - x[i], y[rows] - y[i]
        length = max(float(dist.max()), np.finfo(float).eps)
        weight = np.exp(-.5 * (dist / length) ** 2)
        a, b = dx * np.sqrt(weight[:, None]), dy * np.sqrt(weight[:, None])
        # The dual solve avoids a D × D matrix for high-dimensional inputs.
        covariance = a @ a.T
        regularizer = ridge * np.trace(covariance) / max(1, len(rows))
        if regularizer > 0:
            coefficients = np.linalg.solve(covariance + regularizer * np.eye(len(rows)), b)
            projected[i] = (v[i] @ a.T) @ coefficients
        else:
            projected[i] = v[i] @ np.linalg.pinv(a) @ b
    return VelocityEmbedding(y.copy(), projected, time, basis)


def from_anndata(result, adata, *, use_rep="auto", time_key="time", basis="umap",
                 time=None, time_mapping=None, n_neighbors=30, ridge=1e-3):
    """Read an AnnData without sorting or mutating rows or fitting preprocessing."""
    key = basis if basis.startswith("X_") else "X_" + basis
    if key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{key!r}]; provide an existing embedding.")
    from .anndata_io import resolve_representation, observation_times
    use_rep = resolve_representation(adata, use_rep)
    points = _matrix(adata.X if use_rep is None else adata.obsm[use_rep], "training representation")
    if len(points) != adata.n_obs:
        raise ValueError("Representation rows must match adata.obs.")
    time = observation_times(adata, time=time, time_key=time_key, time_mapping=time_mapping)
    velocity = result.compute_velocity(points, time=time)
    return project_velocity(points, velocity, adata.obsm[key], time=np.broadcast_to(time, (len(points),)).copy(),
                            basis=basis.removeprefix("X_"), n_neighbors=n_neighbors, ridge=ridge)


def _grid(field, *, grid_size, n_neighbors, smooth, min_mass):
    from scipy.spatial import Delaunay, QhullError
    from sklearn.neighbors import NearestNeighbors
    x, v = field.coordinates, field.velocity
    span = np.ptp(x, axis=0)
    if np.any(span <= 0):
        raise ValueError("Stream/grid plotting requires a non-degenerate two-dimensional embedding.")
    axes = [np.linspace(x[:, j].min() - .03 * span[j], x[:, j].max() + .03 * span[j], grid_size) for j in range(2)]
    xx, yy = np.meshgrid(*axes)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(x))).fit(x)
    distances, rows = nn.kneighbors(points)
    local = NearestNeighbors(n_neighbors=min(3, len(x))).fit(x).kneighbors(x)[0][:, -1]
    spacing = float(np.mean([a[1] - a[0] for a in axes]))
    length = max(float(np.median(local)), spacing) * smooth
    weights = np.exp(-.5 * (distances / length) ** 2)
    mass = weights.sum(axis=1)
    flow = (weights[..., None] * v[rows]).sum(axis=1) / np.maximum(mass[:, None], 1e-30)
    valid = (mass >= min_mass * mass.max()) & (distances[:, 0] <= 2.5 * length)
    try:
        valid &= Delaunay(x).find_simplex(points) >= 0
    except QhullError as exc:
        raise ValueError("Stream/grid plotting requires cells spanning a 2D region.") from exc
    return axes, flow.reshape(grid_size, grid_size, 2), valid.reshape(grid_size, grid_size)


def plot_velocity_embedding(field, *, style="stream", color=None, ax=None,
                            grid_size=45, n_neighbors=30, smooth=1., min_mass=.03,
                            density=1.2, point_size=12, alpha=.55, arrow_color="#263445",
                            cmap="viridis", linewidth=.8, arrow_size=1., title=None,
                            palette=None, labels=False, legend=True, norm=None, colorbar_label=None):
    """Plot a learned velocity field over observed cells; return the Axes.

    color can be numeric values, category labels or a Matplotlib color. Stream
    and grid views smooth local velocities and mask unsupported grid regions.
    With no time override, this is a summary of velocities at the cells' own
    observation times. Subset AnnData to a stage for a stage-specific view.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import is_color_like
    from matplotlib.lines import Line2D
    import pandas as pd
    if style not in {"stream", "grid", "arrows", "cells"}:
        raise ValueError("style must be 'stream', 'grid', 'arrows' or 'cells'.")
    if not isinstance(grid_size, int) or grid_size < 5:
        raise ValueError("grid_size must be an integer of at least five.")
    if not isinstance(n_neighbors, int) or n_neighbors < 2:
        raise ValueError("n_neighbors must be an integer of at least two.")
    if not np.isfinite(smooth) or smooth <= 0 or not 0 <= min_mass <= 1:
        raise ValueError("smooth must be positive and min_mass must be between zero and one.")
    x, v = _matrix(field.coordinates, "embedding", 2), _matrix(field.velocity, "velocity", 2)
    if len(x) != len(v):
        raise ValueError("Embedding and velocity rows must match.")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    if color is None or (isinstance(color, str) and is_color_like(color)):
        ax.scatter(*x.T, c=color or "#63A6CF", s=point_size, alpha=alpha, linewidths=0)
    else:
        values = np.asarray(color)
        if values.shape != (len(x),) or pd.isna(values).any():
            raise ValueError("color must contain one non-missing value per cell.")
        if np.issubdtype(values.dtype, np.number) and not isinstance(getattr(color, "dtype", None), pd.CategoricalDtype):
            scatter = ax.scatter(*x.T, c=values, cmap=cmap, norm=norm, s=point_size, alpha=alpha, linewidths=0)
            bar = ax.figure.colorbar(scatter, ax=ax, shrink=.6)
            if colorbar_label is not None:
                bar.set_label(colorbar_label)
        else:
            categories = pd.Categorical(color)
            names = list(categories.categories)
            codes = categories.codes
            colors = category_colors(names, palette)
            ax.scatter(*x.T, c=[colors[j] for j in codes], s=point_size, alpha=alpha, linewidths=0)
            if legend:
                ax.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[j], label=str(label))
                                   for j, label in enumerate(names) if np.any(codes == j)],
                          frameon=False, loc="best", markerscale=.7)
            if labels:
                annotate_celltypes(ax, x, categories)
    if style == "arrows":
        ax.quiver(*x.T, *v.T, color=arrow_color, angles="xy", scale_units="xy", width=.003, alpha=.8)
    elif style != "cells":
        axes, flow, supported = _grid(field, grid_size=grid_size, n_neighbors=n_neighbors, smooth=smooth, min_mass=min_mass)
        if style == "stream":
            u, w = [np.ma.array(flow[..., j], mask=~supported) for j in range(2)]
            if np.any(supported & (np.linalg.norm(flow, axis=-1) > 1e-14)):
                ax.streamplot(*axes, u, w, density=density, color=arrow_color, linewidth=linewidth, arrowsize=arrow_size)
        else:
            xx, yy = np.meshgrid(*axes)
            take = supported.copy()
            # Sparse grid arrows remain legible without changing the estimated flow.
            take[1::2, :] = False; take[:, 1::2] = False
            ax.quiver(xx[take], yy[take], flow[..., 0][take], flow[..., 1][take], color=arrow_color,
                      angles="xy", scale_units="xy", width=.003)
    ax.set(xlabel=f"{field.basis.upper()} 1", ylabel=f"{field.basis.upper()} 2", title=title or "COATI · learned velocity")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def category_colors(names, palette=None):
    """Resolve stable colors in categorical order, including unused categories."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import is_color_like
    if palette is None:
        cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
        colors = [cmap(i % cmap.N) for i in range(len(names))]
    elif isinstance(palette, dict):
        missing = [name for name in names if name not in palette]
        if missing:
            raise ValueError(f"palette is missing cell types: {missing}.")
        colors = [palette[name] for name in names]
    else:
        colors = list(palette)
        if len(colors) != len(names):
            raise ValueError("palette must provide one color per categorical cell type.")
    if not all(is_color_like(color) for color in colors):
        raise ValueError("palette values must be Matplotlib colors.")
    return colors


def annotate_celltypes(ax, coordinates, celltypes):
    """Place labels on observed cells near each group's coordinate median."""
    import pandas as pd
    categories = pd.Categorical(celltypes)
    for code, name in enumerate(categories.categories):
        points = coordinates[categories.codes == code]
        if not len(points):
            continue
        median = np.median(points, axis=0)
        location = points[np.argmin(np.sum((points - median) ** 2, axis=1))]
        ax.text(*location, str(name), ha="center", va="center", fontsize=9, color="#233342",
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.85), zorder=5)
