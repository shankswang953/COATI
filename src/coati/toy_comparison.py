"""Plot matched Gaussian endpoint transport without prescribing either route."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


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
                       distribution_parameters=None, show_group_means=True):
    import torch
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})
    from sklearn.mixture import GaussianMixture
    if distribution_parameters is None:
        # Legacy compact examples estimate their observed terminal mixture.
        mixture = GaussianMixture(n_components=2, random_state=7).fit(data.primary[-1])
        order = np.argsort(mixture.means_[:,1])
        centers = mixture.means_[order]
    else:
        centers = np.asarray(distribution_parameters['terminal_centers'])
    # Use one common view that includes every observed and predicted point.
    all_points = np.concatenate(list(data.primary) + [p['primary'].reshape(-1,2) for p in predictions.values()])
    lower, upper = float(all_points.min()-.04), float(all_points.max()+.04)
    if (all_points >= 0.).all() and (all_points <= 1.).all():
        lower, upper = max(0.,lower), min(1.,upper)
    if distribution_parameters is not None:
        lower, upper = min(0., lower), max(1., upper)
    limits = (lower,upper)
    xx, yy = np.meshgrid(np.linspace(*limits,121), np.linspace(*limits,121))
    mesh = np.stack([xx,yy],-1)
    with torch.no_grad(): surface = mapping(torch.tensor(mesh,dtype=torch.float32)).numpy()[...,2]
    if distribution_parameters is not None:
        surface = np.where(((mesh >= 0.) & (mesh <= 1.)).all(-1), surface, np.nan)
    fig=plt.figure(figsize=(12,9),layout='constrained')
    paths=np.linspace(0,len(data.primary[0])-1,18,dtype=int)
    colors={'Without sync':'#64748b','With sync':'#0072b2'}
    for col,(name,result) in enumerate(results.items()):
        pred=predictions[name];color=colors[name]
        ax=fig.add_subplot(2,2,col+1)
        ax.contourf(xx,yy,surface,levels=np.linspace(0,float(np.nanmax(surface)),18),cmap='Oranges',alpha=.24)
        if distribution_parameters is None:
            initial_center = data.primary[0].mean(0)
            initial_density = multivariate_normal(initial_center,np.cov(data.primary[0].T)).pdf(mesh)
            terminal_density = np.exp(mixture.score_samples(mesh.reshape(-1,2))).reshape(xx.shape)
        else:
            initial_center = np.asarray(distribution_parameters['initial_center'])
            initial_density = gaussian_density_on_unit_square(mesh, initial_center, distribution_parameters['initial_covariance'])
            terminal_density = sum(gaussian_density_on_unit_square(mesh, c, covariance)
                                   for c, covariance in zip(centers, distribution_parameters['terminal_covariances'])) / 2
        # Contour the full normalized mixture; never draw independent crossing rings.
        for density, style in zip([initial_density, terminal_density], ['--', '-']):
            positive = density[density > 0]
            levels = (np.linspace(positive.min(), positive.max(), 5)[1:-1]
                      if distribution_parameters is not None else density.max()*np.array([.12,.4,.75]))
            ax.contour(xx,yy,density,levels=levels,colors='#334155',linestyles=style,linewidths=.9,alpha=.7)
        for cloud_center in np.vstack([initial_center,centers]):
            ax.scatter(*cloud_center,s=38,color='#334155',zorder=4)
        for index in paths:
            ax.plot(pred['primary'][:,index,0],pred['primary'][:,index,1],color=color,lw=1.15,alpha=.45)
        endpoint_group = np.linalg.norm(pred['primary'][-1,:,None,:]-centers[None,:,:],axis=-1).argmin(1)
        for group in range(2):
            mask = endpoint_group == group
            if show_group_means and mask.any():
                mean=pred['primary'][:,mask].mean(1)
                ax.plot(mean[:,0],mean[:,1],color=color,lw=3,label=f'Terminal {group+1} mean')
        field_lower, field_upper = max(0., lower) + .04, min(1., upper) - .04
        gx,gy=np.meshgrid(np.linspace(field_lower,field_upper,13),np.linspace(field_lower,field_upper,13))
        grid=np.column_stack([gx.ravel(),gy.ravel()])
        velocity=result.compute_velocity(grid,time=.5)
        ax.quiver(grid[:,0],grid[:,1],velocity[:,0],velocity[:,1],color=color,alpha=.32,
                  angles='xy',scale_units='xy',scale=10,width=.0025)
        display_name = f"{name} (Cy = {result.metadata['config']['sync_weight']:g})" if name == 'With sync' else name
        ax.set(title=display_name,xlim=limits,ylim=limits,xlabel='Primary dimension 1',ylabel='Primary dimension 2',aspect='equal')
        ax.spines[['top','right']].set_visible(False)
        ax3=fig.add_subplot(2,2,col+3,projection='3d')
        ax3.plot_surface(xx,yy,surface,cmap='Oranges',alpha=.48,linewidth=0,antialiased=True,rcount=70,ccount=80)
        observed_centers = np.vstack([initial_center, centers])
        with torch.no_grad():
            observed_mapped = mapping(torch.tensor(observed_centers,dtype=torch.float32)).numpy()
        ax3.scatter(observed_mapped[:,0],observed_mapped[:,1],observed_mapped[:,2]+.018,
                    color='#334155',s=22,depthshade=False,zorder=7)
        mapped=pred['secondary']
        for index in paths:
            ax3.plot(mapped[:,index,0],mapped[:,index,1],mapped[:,index,2]+.018,color=color,alpha=.5,lw=1.15,zorder=5)
        for group in range(2):
            mask = endpoint_group == group
            if show_group_means and mask.any():
                mean=mapped[:,mask].mean(1)
                ax3.plot(mean[:,0],mean[:,1],mean[:,2]+.022,color=color,lw=3,zorder=6)
        ax3.set(xlim=limits,ylim=limits,zlim=(0,1.08),xlabel='Secondary 1',ylabel='Secondary 2',zlabel='')
        ax3.text2D(.88,.70,'Height',rotation=90,transform=ax3.transAxes,ha='center')
        ax3.view_init(elev=44,azim=-66)
        ax3.set_box_aspect((upper-lower,upper-lower,.8))
        ticks = np.array([0.,.25,.5,.75,1.])
        ticks = ticks[(ticks >= lower) & (ticks <= upper)]
        ax3.set_xticks(ticks);ax3.set_yticks(ticks);ax3.set_zticks([0,.5,1])
        ax3.grid(False)
        for axis in [ax3.xaxis,ax3.yaxis,ax3.zaxis]:axis.pane.fill=False
    fig.suptitle('Gaussian transport · sync vs no sync',fontsize=16)
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
    from ._core.Neural import MLPVectorField
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
