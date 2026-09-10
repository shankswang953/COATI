"""Validated time-indexed inputs. No implicit preprocessing or row reordering."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def _array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.array(x, dtype=np.float32, copy=True)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Each snapshot must be a finite, nonempty cells × features matrix.")
    return x

@dataclass
class TemporalData:
    """Ordered snapshots at explicit numeric times, optionally paired across modalities.

    Secondary rows must represent the same cells in the same order within each time.
    Pairing across different times is neither required nor assumed.
    """
    primary: object
    times: object
    secondary: object = None
    labels: object = None

    def __post_init__(self):
        self.primary = [_array(x) for x in self.primary]
        self.times = np.asarray(self.times, dtype=float)
        if (self.times.ndim != 1 or len(self.times) != len(self.primary)
            or len(self.times) < 2 or not np.isfinite(self.times).all()
            or not (np.diff(self.times) > 0).all()):
            raise ValueError("Provide at least two strictly increasing finite times, one per snapshot.")
        if len({x.shape[1] for x in self.primary}) != 1:
            raise ValueError("Primary feature dimensions must agree at all times.")
        self.labels = list(self.labels) if self.labels is not None else [f"t{i}" for i in range(len(self.times))]
        if len(self.labels) != len(self.times) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Labels must be unique and match the number of times.")
        if self.secondary is not None:
            self.secondary = [_array(x) for x in self.secondary]
            if len(self.secondary) != len(self.primary) or any(len(x) != len(y) for x,y in zip(self.primary,self.secondary)):
                raise ValueError("Paired modalities require identical row counts at each time.")
            if len({x.shape[1] for x in self.secondary}) != 1:
                raise ValueError("Secondary feature dimensions must agree at all times.")

    @classmethod
    def from_dict(cls, primary, *, times, secondary=None):
        """Use mapping insertion order; times explicitly define its numeric coordinates."""
        keys = list(primary)
        if secondary is not None and set(secondary) != set(keys):
            raise ValueError("Primary and secondary keys must match.")
        return cls([primary[k] for k in keys], times,
                   None if secondary is None else [secondary[k] for k in keys], keys)

    @classmethod
    def from_npz(cls, path, *, times, keys=None, secondary_path=None):
        """Read numeric arrays only. Explicit keys are recommended for reproducibility."""
        with np.load(path, allow_pickle=False) as f:
            keys = list(f.files) if keys is None else list(keys)
            primary = [f[k] for k in keys]
        secondary = None
        if secondary_path is not None:
            with np.load(secondary_path, allow_pickle=False) as f:
                secondary = [f[k] for k in keys]
        return cls(primary, times, secondary, keys)

    @classmethod
    def from_dataframe(cls, frame, *, time_key, feature_keys, time_order=None, times=None, secondary_keys=None):
        """Split a table without sorting cells; select feature columns explicitly."""
        if frame[time_key].isna().any():
            raise ValueError("Time labels cannot be missing.")
        order = list(time_order) if time_order is not None else sorted(frame[time_key].unique())
        if set(order) != set(frame[time_key].unique()) or len(set(order)) != len(order):
            raise ValueError("time_order must include each observed time label exactly once.")
        if times is None:
            try: times = [float(t) for t in order]
            except (TypeError, ValueError): raise ValueError("Categorical labels require explicit numeric times.") from None
        groups = [frame.loc[frame[time_key] == t] for t in order]
        return cls([g[list(feature_keys)].to_numpy() for g in groups], times,
                   None if secondary_keys is None else [g[list(secondary_keys)].to_numpy() for g in groups],
                   [str(t) for t in order])

    @classmethod
    def from_csv(cls, path, **kwargs):
        import pandas as pd
        return cls.from_dataframe(pd.read_csv(path), **kwargs)

    @classmethod
    def from_anndata(cls, adata, *, time_key, obsm_key=None, secondary_obsm_key=None, time_order=None, times=None):
        """Use X or an explicit embedding. Both modalities share adata.obs row identities."""
        import pandas as pd
        x = _array(adata.X if obsm_key is None else adata.obsm[obsm_key])
        frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
        features = list(frame)
        frame["__time__"] = adata.obs[time_key].to_numpy()
        secondary_keys = None
        if secondary_obsm_key is not None:
            y = _array(adata.obsm[secondary_obsm_key])
            secondary_keys = [f"y{i}" for i in range(y.shape[1])]
            frame[secondary_keys] = y
        return cls.from_dataframe(frame, time_key="__time__", feature_keys=features,
                                  secondary_keys=secondary_keys, time_order=time_order, times=times)

    @classmethod
    def from_h5ad(cls, path, **kwargs):
        import anndata
        return cls.from_anndata(anndata.read_h5ad(path), **kwargs)

    def save_npz(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        keys = [f"t{i}" for i in range(len(self.times))]
        np.savez(directory / "primary.npz", **dict(zip(keys,self.primary)))
        if self.secondary is not None:
            np.savez(directory / "secondary.npz", **dict(zip(keys,self.secondary)))
        return keys
