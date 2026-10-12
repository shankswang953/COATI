"""Small deterministic synthetic examples; no biological data downloads."""
import numpy as np
from .data import TemporalData

def gaussian(n=128, seed=0):
    """Gaussian translation with three observed times."""
    rng=np.random.default_rng(seed)
    times=[0., .5, 1.]
    return TemporalData([rng.normal(0,.06,(n,2))+[.2+.5*t,.3+.2*t] for t in times],times)

def split(n=128, seed=0):
    """One cloud splits into two; terminal sample count doubles (also usable for UOT)."""
    rng=np.random.default_rng(seed)
    x=rng.normal(0,.055,(n,2))+[.25,.5]
    y=np.concatenate([rng.normal(0,.055,(n,2))+[.75,.25],rng.normal(0,.055,(n,2))+[.75,.75]])
    return TemporalData([x,y],[0.,1.])

def paired(n=128, seed=0):
    """Paired 2D primary and nonlinear 3D secondary snapshots."""
    data=gaussian(n,seed)
    data=TemporalData(data.primary,data.times,[np.column_stack([x, .3*np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])]) for x in data.primary])
    return data

def gaussian_bump_map(*, height=1.5, width=.28, center=(0., .12)):
    """Return the known differentiable 2D-to-3D map for the Gaussian sync toy.

    The returned torch.nn.Module accepts forward(points, t=None). A fixed
    Gaussian height adds secondary geometry without prescribing a primary path.
    No map parameters are fitted; coordinates remain in raw toy units.
    """
    import torch
    from torch import nn
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(height) or height <= 0 or not np.isfinite(width) or width <= 0:
        raise ValueError("Use a finite 2D center and positive finite bump height/width.")

    class GaussianBumpMap(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
            self.register_buffer("height", torch.tensor(height, dtype=torch.float32))
            self.register_buffer("width", torch.tensor(width, dtype=torch.float32))

        def forward(self, points, t=None):
            squared_radius = ((points-self.center)/self.width).square().sum(-1, keepdim=True)
            altitude = self.height * torch.exp(-.5*squared_radius)
            return torch.cat([points, altitude], dim=-1)

    return GaussianBumpMap()


def gaussian_sync(n=128, seed=7, *, height=1.5, width=.28, center=(0., .12)):
    """Endpoint Gaussian clouds paired through a known Gaussian-bump surface.

    Only t=0 and t=1 are observed: no intermediate detour is given to training.
    The two primary clouds have different covariance shapes and orientations.
    Each secondary row is the exact image of its corresponding primary row.
    gaussian_bump_map with the same parameters supplies the known map for fitting.
    Secondary clouds are pushforwards, not Gaussians in three dimensions.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer of at least two.")
    rng = np.random.default_rng(seed)
    primary = []
    for mean, std, angle in [((-1., -.4), (.065, .045), 25.), ((1., .4), (.09, .05), -20.)]:
        theta = np.deg2rad(angle)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        primary.append(rng.normal(size=(n,2)) * std @ rotation.T + mean)
    mapping = gaussian_bump_map(height=height, width=width, center=center)
    import torch
    with torch.no_grad():
        secondary = [mapping(torch.tensor(x, dtype=torch.float32)).numpy() for x in primary]
    return TemporalData(primary, [0.,1.], secondary)
