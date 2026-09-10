"""Configuration independent of command-line parsers and experiment folders."""
import copy
import json
from pathlib import Path

class Config:
    """Keep the original research parameter names; defaults target small CPU examples.

    Additional engine parameters may be supplied using ``extra={...}``.
    No normalization is applied by default. Call fit with normalization parameter paths
    to reuse the research pipeline's normalization exactly.
    """
    def __init__(self, *, extra=None, **kwargs):
        defaults = json.loads(Path(__file__).with_name("_defaults.json").read_text())
        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise TypeError(f"Unknown config fields: {sorted(unknown)}. Advanced engine fields belong in extra.")
        defaults.update(kwargs)
        defaults.update(extra or {})
        self.values = defaults
        for key in ("niters", "num_samples", "median_n_sample", "floor_n_sample", "hidden_dim"):
            if not isinstance(defaults[key], int) or defaults[key] < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if defaults["dt"] <= 0 or defaults["lr"] <= 0:
            raise ValueError("dt and lr must be positive.")
        if not 0 <= defaults["sync_weight"] <= 1:
            raise ValueError("sync_weight must be between zero and one.")
        if defaults["support_points"]:
            raise ValueError("The public API supports snapshot data only; support_points requires the advanced engine API.")

    def to_dict(self):
        return copy.deepcopy(self.values)

    def save(self, path):
        Path(path).write_text(json.dumps(self.values, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(extra=json.loads(Path(path).read_text()))

    def __repr__(self):
        return f"Config({self.values!r})"
