"""Configuration independent of command-line parsers and experiment folders."""
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MapConfig:
    """Paired map fitting and validation, separate from trajectory settings.

    Quality thresholds are configurable diagnostics, not universal acceptance criteria.
    Validation cells are withheld within each supplied stage, not at new stages.
    """
    hidden_dim: int = 128
    n_layers: int = 3
    time_hidden_dim: int = 32
    epochs: int = 300
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    patience: int = 30
    eval_samples: int = 256
    eval_repeats: int = 5
    min_validation_cells: int = 20
    max_normalized_mse: float = 0.5
    max_sinkhorn_ratio: float = 5.0
    quality_action: str = "warn"

    def __post_init__(self):
        for key in ("hidden_dim", "n_layers", "time_hidden_dim", "epochs", "batch_size",
                    "patience", "eval_samples", "eval_repeats", "min_validation_cells"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"MapConfig.{key} must be a positive integer.")
        if self.n_layers < 2:
            raise ValueError("MapConfig.n_layers must be at least two.")
        for key in ("lr", "max_normalized_mse", "max_sinkhorn_ratio"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"MapConfig.{key} must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("MapConfig.weight_decay must be finite and nonnegative.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("MapConfig.validation_fraction must be between zero and one.")
        if self.quality_action not in {"warn", "error"}:
            raise ValueError("MapConfig.quality_action must be 'warn' or 'error'.")

    def to_dict(self):
        return asdict(self)

class Config:
    """Keep the original research parameter names; defaults target small CPU examples.

    Additional engine parameters may be supplied using ``extra={...}``.
    No normalization is applied by default. Call fit with normalization parameter paths
    to reuse the research pipeline's normalization exactly.
    """
    def __init__(self, *, multimodal=None, balanced=None, extra=None, **kwargs):
        aliases = {"iterations": "niters", "learning_rate": "lr", "batch_size": "num_samples",
                   "secondary_weight": "sync_weight", "matching_weight": "pdf_coefficient",
                   "manifold_weight": "density_coefficient", "energy_weight": "energy_coefficient",
                   "growth_weight": "alpha_growth", "mass_weight": "mass_coefficient"}
        kwargs = dict(kwargs)
        extra = dict(extra or {})
        for alias, original in aliases.items():
            if alias in kwargs:
                if original in kwargs or original in extra:
                    raise ValueError(f"Supply either {alias} or {original}, not both.")
                kwargs[original] = kwargs.pop(alias)
        supplied = {**kwargs, **extra}
        for name, mode in (("multimodal", multimodal), ("balanced", balanced)):
            if mode is not None and not isinstance(mode, bool):
                raise TypeError(f"{name} must be True or False.")
        if multimodal is not None:
            if "sync_loss" in supplied and supplied["sync_loss"] != multimodal:
                raise ValueError("multimodal conflicts with sync_loss.")
            kwargs["sync_loss"] = multimodal
            if "sync_weight" not in supplied:
                kwargs["sync_weight"] = .35 if multimodal else 0.
        if balanced is not None:
            if "unbalancedModel" in supplied and supplied["unbalancedModel"] != (not balanced):
                raise ValueError("balanced conflicts with unbalancedModel.")
            kwargs["unbalancedModel"] = not balanced
            # Only explicit mode selection supplies the growth/mass preset.
            for key, value in {"alpha_growth": 0. if balanced else 1.,
                               "mass_loss": not balanced,
                               "mass_coefficient": 0. if balanced else 1.}.items():
                if key not in supplied:
                    kwargs[key] = value
        # Old W2-only saved configs contain disabled MMD flags. Drop only False;
        # never silently reinterpret a configuration that requested MMD matching.
        kwargs = dict(kwargs)
        extra = dict(extra or {})
        for fields in (kwargs, extra):
            for key in ("mmd_loss", "sync_mmd_loss"):
                if key in fields:
                    if fields.pop(key) is not False:
                        raise ValueError(f"{key} has been removed; matching uses W2 (Sinkhorn, p=2).")
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

    @property
    def multimodal(self):
        """Whether synchronization through a secondary modality is enabled."""
        return bool(self.values["sync_loss"])

    @property
    def balanced(self):
        """Whether population mass is conserved by the trajectory model."""
        return not bool(self.values["unbalancedModel"])

    def with_parameters(self, **kwargs):
        """Return a new config with clear names or original engine names overridden."""
        aliases = {"iterations": "niters", "learning_rate": "lr", "batch_size": "num_samples",
                   "secondary_weight": "sync_weight", "matching_weight": "pdf_coefficient",
                   "manifold_weight": "density_coefficient", "energy_weight": "energy_coefficient",
                   "growth_weight": "alpha_growth", "mass_weight": "mass_coefficient"}
        kwargs = dict(kwargs)
        advanced = dict(kwargs.pop("extra", {}) or {})
        values = self.to_dict()
        values.update(advanced)
        for key in kwargs:
            values.pop(aliases.get(key, key), None)
        if "multimodal" in kwargs:
            values.pop("sync_loss", None)
            if (bool(kwargs["multimodal"]) != self.multimodal
                and not {"secondary_weight", "sync_weight"} & kwargs.keys()
                and "sync_weight" not in advanced):
                values.pop("sync_weight", None)
        if "balanced" in kwargs:
            values.pop("unbalancedModel", None)
            if bool(kwargs["balanced"]) != self.balanced:
                for key in ("alpha_growth", "mass_loss", "mass_coefficient"):
                    if key not in advanced:
                        values.pop(key, None)
        return Config(extra=values, **kwargs)

    def to_dict(self):
        return copy.deepcopy(self.values)

    def save(self, path):
        Path(path).write_text(json.dumps(self.values, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(extra=json.loads(Path(path).read_text()))

    def __repr__(self):
        return f"Config({self.values!r})"
