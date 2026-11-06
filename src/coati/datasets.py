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

def gaussian_bump_map(*, height=1., width=.10, center=(.5, .5)):
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


def gaussian_sync(n=128, seed=7, *, height=1., width=.10, center=(.5, .5),
                  terminal_centers=((.50, .72), (.72, .50)),
                  initial_center=(.30, .30), initial_covariance=None,
                  terminal_covariances=None, terminal_mass=1.):
    """One Gaussian source and two terminal Gaussians on a known bump surface.

    Only t=0 and t=1 are observed: no intermediate detour is given to training.
    The paper-inspired geometry has two nearby destinations beyond the bump.
    terminal_centers can adjust their separation without prescribing trajectories.
    Compact, distinct covariances follow the original toy data generator; all
    observations are sampled inside [0,1]^2. No intermediate path is specified.
    n is the initial cell count. terminal_mass sets the final/initial count
    ratio, representing the relative terminal mass in unbalanced training.
    Its product with n must be an integer of at least two. Odd terminal
    sample counts approximate the equal mixture with counts differing by one.
    Explicit covariances describe Gaussian kernels BEFORE truncation to the
    unit square; the covariance of the resulting observations is different.
    None retains the earlier compact tutorial covariances. The manuscript
    reference notebook supplies identity covariances and the map width explicitly.
    Each secondary row is the exact image of its corresponding primary row.
    gaussian_bump_map with the same parameters supplies the known map for fitting.
    Secondary clouds are pushforwards, not Gaussians in three dimensions.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer of at least two.")
    terminal_centers = np.asarray(terminal_centers, dtype=float)
    if terminal_centers.shape != (2,2) or not np.isfinite(terminal_centers).all() or not ((terminal_centers >= 0.) & (terminal_centers <= 1.)).all():
        raise ValueError("terminal_centers must contain two finite 2D centers inside [0,1]^2.")
    initial_center = np.asarray(initial_center, dtype=float)
    if initial_center.shape != (2,) or not np.isfinite(initial_center).all() or not ((initial_center >= 0.) & (initial_center <= 1.)).all():
        raise ValueError("initial_center must be finite and inside [0,1]^2.")
    terminal_mass = float(terminal_mass)
    count = n * terminal_mass
    if not np.isfinite(count) or count < 2 or not np.isclose(count, round(count), rtol=0, atol=1e-8):
        raise ValueError("terminal_mass * n must be an integer of at least two.")
    n_terminal = int(round(count))
    initial_covariance = np.asarray([[.0042, .0002], [.0002, .0038]] if initial_covariance is None else initial_covariance, dtype=float)
    terminal_covariances = np.asarray([[[.0030, .0001], [.0001, .0030]], [[.0032, -.0001], [-.0001, .0032]]] if terminal_covariances is None else terminal_covariances, dtype=float)
    if initial_covariance.shape != (2,2) or terminal_covariances.shape != (2,2,2):
        raise ValueError("Provide one 2x2 initial covariance and two 2x2 terminal covariances.")
    for covariance in [initial_covariance, *terminal_covariances]:
        if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("Gaussian covariances must be finite, symmetric and positive definite.")
    rng = np.random.default_rng(seed)

    def cloud(count, mean, covariance):
        # Rejection sampling keeps the Gaussian observations inside the unit square.
        samples = []
        remaining = count
        while remaining:
            batch = rng.multivariate_normal(mean, covariance, size=max(2*remaining, 128))
            valid = batch[((batch >= 0.) & (batch <= 1.)).all(axis=1)][:remaining]
            samples.append(valid)
            remaining -= len(valid)
        return np.concatenate(samples)

    initial = cloud(n, initial_center, initial_covariance)
    terminal = np.concatenate([
        cloud(n_terminal//2, terminal_centers[0], terminal_covariances[0]),
        cloud(n_terminal-n_terminal//2, terminal_centers[1], terminal_covariances[1]),
    ])
    primary = [initial, terminal]
    mapping = gaussian_bump_map(height=height, width=width, center=center)
    import torch
    with torch.no_grad():
        secondary = [mapping(torch.tensor(x, dtype=torch.float32)).numpy() for x in primary]
    return TemporalData(primary, [0.,1.], secondary)
