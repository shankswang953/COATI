"""Plot matched Gaussian endpoint transport without prescribing either route."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def plot_gaussian_sync(data, mapping, results, predictions):
    import torch
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})
    xx, yy = np.meshgrid(np.linspace(-1.28,1.32,80), np.linspace(-.83,.85,70))
    mesh = np.stack([xx,yy],-1)
    with torch.no_grad(): surface = mapping(torch.tensor(mesh,dtype=torch.float32)).numpy()[...,2]
    fig=plt.figure(figsize=(12,9),layout='constrained')
    paths=np.linspace(0,len(data.primary[0])-1,18,dtype=int)
    colors={'Without sync':'#64748b','With sync':'#0072b2'}
    for col,(name,result) in enumerate(results.items()):
        pred=predictions[name];color=colors[name]
        ax=fig.add_subplot(2,2,col+1)
        ax.contourf(xx,yy,surface,levels=np.linspace(0,float(surface.max()),18),cmap='Oranges',alpha=.24)
        for points,cloud_color in zip(data.primary,['#334155','#334155']):
            density=multivariate_normal(points.mean(0),np.cov(points.T)).pdf(mesh)
            ax.contour(xx,yy,density,levels=density.max()*np.array([.12,.4,.75]),colors=cloud_color,linewidths=.9,alpha=.7)
        for index in paths:
            ax.plot(pred['primary'][:,index,0],pred['primary'][:,index,1],color=color,lw=1.15,alpha=.45)
        mean=pred['primary'].mean(1)
        ax.plot(mean[:,0],mean[:,1],color=color,lw=3,label='Mean learned trajectory')
        ax.scatter(*data.primary[0].mean(0),s=38,color='#334155',zorder=4)
        ax.scatter(*data.primary[-1].mean(0),s=38,color='#334155',zorder=4)
        gx,gy=np.meshgrid(np.linspace(-1.05,1.05,13),np.linspace(-.62,.64,9))
        grid=np.column_stack([gx.ravel(),gy.ravel()])
        velocity=result.compute_velocity(grid,time=.5)
        ax.quiver(grid[:,0],grid[:,1],velocity[:,0],velocity[:,1],color=color,alpha=.32,
                  angles='xy',scale_units='xy',scale=13,width=.0025)
        ax.set(title=name,xlim=(-1.28,1.32),ylim=(-.83,.85),xlabel='Primary dimension 1',ylabel='Primary dimension 2',aspect='equal')
        ax.spines[['top','right']].set_visible(False)
        ax3=fig.add_subplot(2,2,col+3,projection='3d')
        ax3.plot_surface(xx,yy,surface,cmap='Oranges',alpha=.48,linewidth=0,antialiased=True,rcount=70,ccount=80)
        mapped=pred['secondary']
        for index in paths:
            ax3.plot(mapped[:,index,0],mapped[:,index,1],mapped[:,index,2]+.018,color=color,alpha=.5,lw=1.15,zorder=5)
        mean=mapped.mean(1)
        ax3.plot(mean[:,0],mean[:,1],mean[:,2]+.022,color=color,lw=3,zorder=6)
        ax3.set(xlim=(-1.28,1.32),ylim=(-.83,.85),zlim=(0,1.6),xlabel='Secondary 1',ylabel='Secondary 2',zlabel='Height')
        ax3.view_init(elev=44,azim=-66)
        ax3.set_box_aspect((2.6,1.65,1.3))
        ax3.set_xticks([-1,0,1]);ax3.set_yticks([-.5,0,.5]);ax3.set_zticks([0,.5,1,1.5])
        ax3.grid(False)
        for axis in [ax3.xaxis,ax3.yaxis,ax3.zaxis]:axis.pane.fill=False
    fig.suptitle('Gaussian transport · synchronization changes the route',fontsize=16)
    return fig


def plot_bump_height(predictions):
    fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
    for (name,pred),color in zip(predictions.items(),['#64748b','#0072b2']):
        h=pred['secondary'][...,2]
        ax.plot(pred['times'],h.mean(1),color=color,lw=2.5,label=name)
        lo,hi=np.quantile(h,[.25,.75],axis=1)
        ax.fill_between(pred['times'],lo,hi,color=color,alpha=.13)
    ax.set(xlabel='Time',ylabel='Mapped bump height',title='Same map, different learned routes')
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
    """Describe endpoint fit and mapped path cost on one common prediction grid.

    Secondary kinetic energy is 0.5 * sum ||Delta T(x)||^2 / Delta t, averaged
    across paths. Both models are evaluated through the same known toy map,
    including the model that did not use secondary information during fitting.
    These raw-unit finite-grid diagnostics are not a biological benchmark.
    """
    import pandas as pd
    rows = []
    for name, result in results.items():
        pred = predictions[name]
        delta_t = np.diff(pred['times'])
        mapped = pred['secondary']
        energy = .5 * (np.square(np.diff(mapped,axis=0)).sum(-1)/delta_t[:,None]).sum(0).mean()
        row = {'Model': name, 'Primary endpoint Sinkhorn': result.evaluate()['endpoint_sinkhorn_after'],
               'Primary kinetic energy': float(pred['energy'][-1].mean()),
               'Secondary kinetic energy': float(energy),
               'Median maximum bump height': float(np.median(mapped[...,2].max(0)))}
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')
