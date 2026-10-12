"""COATI: Cross-Omics Aligned Trajectory Inference."""
from .data import TemporalData
from .config import Config, MapConfig
__version__ = "0.1.0"

def fit(*args, **kwargs):
    """Train a trajectory model; see coati.training.fit."""
    from .training import fit as _fit
    return _fit(*args, **kwargs)

def load_result(*args, **kwargs):
    from .result import Result
    return Result.load(*args, **kwargs)

from . import datasets


def train_vector_field(*args, **kwargs):
    """Train with explicit multimodal/balanced controls; see coati.training."""
    from .training import train_vector_field as _train
    return _train(*args, **kwargs)
