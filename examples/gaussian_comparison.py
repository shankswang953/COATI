"""Plot matched Gaussian endpoint transport without prescribing either route."""
import numpy as np
import matplotlib.pyplot as plt


def gaussian_sync_example(*, balanced=True, secondary_weight=None,
                          iterations=None, warmup_iterations=None, seed=None):
    """Prepare the Gaussian tutorial from its repository example configuration.

    Numeric defaults live in gaussian_sync.json, shared by the notebook
    and command-line example. Both comparisons use the same endpoint data,
    primary-only warm-up checkpoint, and refinement budget.
    """
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from coati import Config
    from toy_data import gaussian_sync, gaussian_bump_map
    preset = json.loads(Path(__file__).with_name('gaussian_sync.json').read_text())
    for key, value in [('secondary_weight', secondary_weight),
                       ('refinement_iterations', iterations),
                       ('warmup_iterations', warmup_iterations), ('seed', seed)]:
        if value is not None:
            preset[key] = value
    mass = 1 if balanced else 2
    data = gaussian_sync(n=preset['n_cells'], seed=preset['seed'],
                         terminal_mass=mass, **preset['bump'], **preset['distribution'])
    mapping = gaussian_bump_map(**preset['bump'])
    base = Config(multimodal=False, balanced=balanced,
                  iterations=preset['refinement_iterations'], seed=preset['seed'],
                  mass_weight=0. if balanced else preset['mass_weight'],
                  global_mass=not balanced, **preset['training'])
    return SimpleNamespace(
        data=data, mapping=mapping, distribution=preset['distribution'],
        terminal_mass=mass, settings=preset,
        configs={'Without sync': base, 'With sync': base.with_parameters(
            multimodal=True, secondary_weight=preset['secondary_weight'])},
        warmup_config=base.with_parameters(iterations=preset['warmup_iterations'],
                                          learning_rate=preset['warmup_learning_rate']),
        times=np.linspace(data.times[0], data.times[-1], preset['prediction_points']),
    )


def gaussian_density_on_unit_square(points, center, covariance):
    """Evaluate N_X from a Gaussian kernel, normalized on [0,1]^2.

    Fixed-grid quadrature evaluates the normalizer; no sample-fitted mixture
    or inferred component assignments enter this reference density.
    """
    from scipy.integrate import trapezoid
    center, covariance = np.asarray(center, float), np.asarray(covariance, float)
    inverse = np.linalg.inv(covariance)
    def kernel(q):
        offset = np.asarray(q) - center
        return np.exp(-.5 * np.einsum('...i,ij,...j->...', offset, inverse, offset))
    grid = np.linspace(0., 1., 401)
    xx, yy = np.meshgrid(grid, grid)
    normalizer = trapezoid(trapezoid(kernel(np.stack([xx, yy], -1)), grid, axis=1), grid)
    query = np.asarray(points)
    inside = ((query >= 0.) & (query <= 1.)).all(axis=-1)
    return np.where(inside, kernel(query) / normalizer, 0.)


def _normalized_path_weights(prediction):
    shape = np.asarray(prediction['primary']).shape[:2]
    if 'log_weights' not in prediction:
        return np.full(shape, 1 / shape[1])
    logw = np.asarray(prediction['log_weights']).reshape(shape)
    weights = np.exp(logw - logw.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def plot_gaussian_sync(data, mapping, results, predictions, *,
                       distribution_parameters=None, show_group_means=False,
                       show_vector_field=False):
    """Show the full learned trajectory population against secondary geometry.

    Rows compare methods; columns show primary and mapped secondary states.
    Grey points are observed endpoints. Black level sets belong to the bump,
    not to endpoint densities. Every particle is drawn without branch selection,
    clipping or route smoothing. Optional group means retain the legacy display.
    Optional arrows
    show the instantaneous primary field at t=0.5.
    """
    import torch
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})
    all_points = np.concatenate(list(data.primary) + [
        np.asarray(p['primary']).reshape(-1, 2) for p in predictions.values()])
    lower = min(0., float(all_points.min()) - .025)
    upper = max(1., float(all_points.max()) + .025)
    # Keep the unit-square view when every state lies inside the domain.
    if (all_points >= 0.).all() and (all_points <= 1.).all():
        lower, upper = 0., 1.
    grid = np.linspace(0., 1., 181)
    xx, yy = np.meshgrid(grid, grid)
    mesh = np.stack([xx, yy], -1)
    with torch.no_grad():
        surface = mapping(torch.tensor(mesh, dtype=torch.float32)).cpu().numpy()[..., 2]
    observed = np.concatenate(data.primary)
    fig = plt.figure(figsize=(9.4, 8.0))
    fig.subplots_adjust(left=.08, right=.99, bottom=.07, top=.94, wspace=.10, hspace=.19)
    palette = ['#8878ae', '#1688b1']
    for row, (name, result) in enumerate(results.items()):
        pred = predictions[name]
        primary, mapped = np.asarray(pred['primary']), np.asarray(pred['secondary'])
        if primary.shape[:2] != mapped.shape[:2]:
            raise ValueError("Primary and secondary predictions must share times and particles.")
        if show_group_means:
            if distribution_parameters is None:
                from sklearn.mixture import GaussianMixture
                centers = GaussianMixture(n_components=2, random_state=7).fit(data.primary[-1]).means_
            else:
                centers = np.asarray(distribution_parameters['terminal_centers'])
            groups = np.linalg.norm(primary[-1, :, None] - centers[None], axis=-1).argmin(1)
        colour = palette[row % len(palette)]
        title = (f"Sync · Cy = {result.metadata['config']['sync_weight']:g}"
                 if result.metadata['config']['sync_loss'] else 'No sync')
        ax = fig.add_subplot(len(results), 2, 2*row+1)
        ax.scatter(observed[:, 0], observed[:, 1], s=3, color='#9ca3af',
                   alpha=.30, linewidths=0, rasterized=True, zorder=1)
        # Reference figure contours encode the central Gaussian bump.
        ax.contour(xx, yy, surface, levels=np.asarray([.05, .3, .7])*surface.max(),
                   colors='#343a40', linewidths=.8, alpha=.8, zorder=3)
        lines = LineCollection(primary.transpose(1, 0, 2), colors=colour,
                               linewidths=.55, alpha=.17, rasterized=True, zorder=2)
        ax.add_collection(lines)
        if show_group_means:
            for group in range(2):
                mask = groups == group
                if mask.any():
                    mean = primary[:, mask].mean(1)
                    ax.plot(mean[:, 0], mean[:, 1], color=colour, linewidth=2, zorder=4)
        if show_vector_field:
            gx, gy = np.meshgrid(np.linspace(.08, .92, 11), np.linspace(.08, .92, 11))
            points = np.column_stack([gx.ravel(), gy.ravel()])
            velocity = result.compute_velocity(points, time=.5)
            ax.quiver(points[:, 0], points[:, 1], velocity[:, 0], velocity[:, 1],
                      color=colour, alpha=.5, angles='xy', scale_units='xy', scale=12,
                      width=.0025, zorder=4)
        ax.set(xlim=(lower, upper), ylim=(lower, upper), aspect='equal',
               xlabel='$x_1$', ylabel='$x_2$')
        if row == 0:
            ax.set_title('Primary', fontsize=14, pad=12)
        ax.text(.03, .97, title, transform=ax.transAxes, va='top', color=colour,
                fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
        ax.set_xticks([0., .5, 1.]); ax.set_yticks([0., .5, 1.])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)

        ax3 = fig.add_subplot(len(results), 2, 2*row+2, projection='3d', computed_zorder=False)
        ax3.plot_surface(xx, yy, surface, color='#cbd0d6', alpha=.23, linewidth=0,
                         antialiased=True, rcount=90, ccount=90, zorder=1)
        lifted_lines = Line3DCollection(mapped.transpose(1, 0, 2), colors=colour,
                                        linewidths=.55, alpha=.21, rasterized=True, zorder=3)
        ax3.add_collection3d(lifted_lines)
        if show_group_means:
            for group in range(2):
                mask = groups == group
                if mask.any():
                    mean = mapped[:, mask].mean(1)
                    ax3.plot(mean[:, 0], mean[:, 1], mean[:, 2], color=colour, linewidth=2, zorder=4)
        ax3.set(xlim=(lower, upper), ylim=(lower, upper), zlim=(0., float(surface.max())*1.05))
        if row == 0:
            ax3.set_title('Secondary', fontsize=14, pad=12)
        ax3.set_xticks([0., 1.]); ax3.set_yticks([0., 1.]); ax3.set_zticks([0., float(surface.max())])
        ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
        ax3.tick_params(pad=-1)
        ax3.view_init(elev=38, azim=-58)
        ax3.set_box_aspect((1., 1., .8), zoom=.97)
        ax3.grid(False)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor('#d8dce0')
    return fig


def plot_bump_height(predictions):
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    for (name,pred),color in zip(predictions.items(),['#64748b','#0072b2']):
        h=pred['secondary'][...,2]
        weights = _normalized_path_weights(pred)
        ax.plot(pred['times'],(h * weights).sum(1),color=color,lw=2.5,label=name)
        quantiles = []
        for values, w in zip(h, weights):
            order = np.argsort(values)
            quantiles.append(np.interp([.25, .75], np.cumsum(w[order]) - .5*w[order], values[order]))
        lo, hi = np.asarray(quantiles).T
        ax.fill_between(pred['times'],lo,hi,color=color,alpha=.13)
    ax.set(xlabel='Time',ylabel='Mapped bump height',title='Mapped bump height along the learned paths')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False)
    return fig


def make_shared_initial_checkpoint(config, path, *, dimension=2):
    """Save identical starting weights for a controlled comparison.

    The training engine consumes random numbers for modality-specific diagnostics
    before constructing its network. A common weight-only warm start guarantees
    identical model initialization; it does not promise identical minibatch draws.
    Existing files are never overwritten, and the caller's RNG state is restored.
    """
    from pathlib import Path
    import torch
    from coati._core.Neural import MLPVectorField
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Initial checkpoint already exists: {path}")
    values = config.to_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(values['seed'])
        model = MLPVectorField(dim=dimension, hidden_dim=values['hidden_dim'],
                               n_layers=values['n_layers'], activation=values['activation'],
                               unbalanced=values['unbalancedModel'], alpha_growth=values['alpha_growth'])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'func_state_dict': model.state_dict(), 'iteration': 0}, path)
    return path


def compare_gaussian_runs(results, predictions):
    """Compare shape matching, mass and kinetic action on a common grid.

    Unbalanced actions use raw left-endpoint masses and sum over particles.
    Endpoint Sinkhorn uses normalized mass for shape; total mass is separate.
    Balanced actions average the equal-weight paths. Coordinates are converted
    to training units for the primary matching diagnostic.
    """
    import pandas as pd
    import torch
    from geomloss import SamplesLoss
    rows = []
    for name, result in results.items():
        pred = predictions[name]
        delta_t = np.diff(pred['times'])
        mapped = pred['secondary']
        per_path = .5 * np.square(np.diff(mapped,axis=0)).sum(-1) / delta_t[:,None]
        if 'log_weights' in pred:
            weights = np.exp(np.asarray(pred['log_weights']).reshape(per_path.shape[0]+1, per_path.shape[1]))
            energy = (per_path * weights[:-1]).sum()
            primary_energy = pred['energy'][-1].sum()
            terminal_mass = weights[-1].sum()
        else:
            energy = per_path.sum(0).mean()
            primary_energy = pred['energy'][-1].mean()
            terminal_mass = 1.
        scale = result.metadata['primary_scale']
        endpoint = torch.as_tensor(pred['primary'][-1]/scale, dtype=torch.float32)
        target = torch.as_tensor(result.data.primary[-1]/scale, dtype=torch.float32)
        source_weights = torch.as_tensor(_normalized_path_weights(pred)[-1], dtype=torch.float32)
        target_weights = torch.full((len(target),),1/len(target))
        loss = SamplesLoss('sinkhorn', p=2, blur=result.metadata['blur_primary'][-1], backend='tensorized')
        with torch.no_grad(): shape_loss = float(loss(source_weights, endpoint, target_weights, target))
        row = {'Model': name, 'Primary endpoint Sinkhorn': shape_loss,
               'Primary kinetic energy': float(primary_energy),
               'Secondary kinetic energy': float(energy),
               'Predicted terminal mass': float(terminal_mass),
               'Target terminal mass': len(result.data.primary[-1])/len(result.data.primary[0]),
               'Median maximum bump height': float(np.median(mapped[...,2].max(0))),
               'States outside unit square': float(((pred['primary'] < 0.) | (pred['primary'] > 1.)).any(-1).mean())}
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')
