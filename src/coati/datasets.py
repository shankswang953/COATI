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
