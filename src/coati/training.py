"""Public training adapter around the preserved research pipeline."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import copy
import json
import uuid
from .config import Config, MapConfig
from .data import TemporalData


def fit(data, config=None, *, output_dir=None, T_model=None, map_model=None,
        map_config=None,
        primal_norm_path=None, sec_norm_path=None, warm_start_path=None,
        metric_primal=None, metric_secondary=None, verbose=False):
    """Fit validated snapshots and return a portable Result.

    Output directories must be empty; repeated notebook runs never erase previous logs.
    Normalization is opt-in, using the original engine's saved scale dictionaries.
    With synchronization enabled, a FiLM map is fitted first unless a map is supplied.
    ``Config(map_type="mlp")`` fits an ordinary MLP without time input. Kernel modes
    remain explicit alternatives. ``map_config`` controls paired neural-map training.
    A supplied ``T_model`` accepts ``forward(q)`` or ``forward(q, t)`` in training
    coordinates and is frozen on a copy. Only supplied snapshots are used; omitted
    stages are never loaded for either map fitting or trajectory fitting.
    """
    if not isinstance(data, TemporalData):
        raise TypeError("Wrap inputs with TemporalData or one of its from_* constructors.")
    cfg = config or Config()
    values = cfg.to_dict()
    if map_config is not None and not isinstance(map_config, MapConfig):
        raise TypeError("map_config must be a MapConfig.")
    if T_model is not None and map_model is not None:
        raise ValueError("Supply only one of T_model and map_model.")
    map_kind = str(values["map_type"]).lower()
    builtin_map = values["sync_loss"] and T_model is None and map_model is None and map_kind in {"film", "mlp"}
    if map_config is not None and not builtin_map:
        raise ValueError("map_config is only used when fitting a built-in FiLM or MLP map.")
    if values["sync_loss"] and T_model is None and map_model is None and not builtin_map and values["map_type"] not in {"kernel", "accurate_kernel"}:
        raise ValueError("Without a supplied map, map_type must be 'film', 'mlp', 'kernel' or 'accurate_kernel'.")
    if min(map(len, data.primary)) < values["num_samples"]:
        raise ValueError("num_samples exceeds the smallest snapshot; choose a smaller batch explicitly.")
    if values["sync_loss"] and data.secondary is None:
        raise ValueError("sync_loss=True requires paired secondary snapshots.")
    if not values["sync_loss"] and (T_model is not None or map_model is not None):
        raise ValueError("External maps require sync_loss=True.")
    import numpy as np
    intervals = np.diff(data.times) / values["dt"]
    if np.any(intervals < 1) or not np.allclose(intervals, np.round(intervals), atol=1e-6):
        raise ValueError("Each interval must be an integer multiple of dt: the preserved energy uses dt explicitly.")
    out = Path(output_dir or Path("outputs") / ("run-" + uuid.uuid4().hex[:10])).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a fresh directory.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir()
    keys = data.save_npz(out / "inputs")
    cfg.save(out / "config.json")
    values.update(data_path=str(out/"inputs/primary.npz"), sync_data_path=str(out/"inputs/secondary.npz"),
                  time_labels=keys, time_points=data.times.tolist(), otdim=data.primary[0].shape[1],
                  train_dir=str(out/"checkpoints"), results_dir=str(out/"logs"))
    # Import scientific backends only when training is requested.
    from ._core.Training import run_training, _resolve_device
    from .result import Result
    args = SimpleNamespace(**values)
    device = _resolve_device(args)
    import torch
    def scale(path):
        value = 1.0 if path is None else float(torch.load(path, map_location="cpu", weights_only=True)["scale"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Normalization scale must be finite and positive.")
        return value
    primary_scale, secondary_scale = scale(primal_norm_path), scale(sec_norm_path)
    from .maps import train_map, freeze_external_map
    if builtin_map:
        if verbose:
            print(f"[COATI] Fitting {map_kind} map using supplied times {data.times.tolist()}.")
        T_model, map_report = train_map(data, kind=map_kind, config=map_config,
                                       directory=out/"map", device=device, seed=values["seed"],
                                       primary_scale=primary_scale, secondary_scale=secondary_scale, verbose=verbose)
        if verbose:
            print(f"[COATI] Map validation: {map_report['status']}. Starting synchronized trajectory fitting.")
    elif T_model is not None:
        T_model = freeze_external_map(T_model, device)
    def frozen(model):
        if model is None: return None
        model = copy.deepcopy(model).to(device).eval()
        for p in model.parameters(): p.requires_grad_(False)
        return model
    map_model = frozen(map_model)
    if T_model is not None:
        with torch.no_grad():
            sample = torch.tensor(data.primary[0][:2] / primary_scale, device=device).unsqueeze(0)
            mapped = T_model(sample, sample.new_full(sample.shape[:-1], float(data.times[0])))
            if mapped.shape != (*sample.shape[:-1], data.secondary[0].shape[1]) or not torch.isfinite(mapped).all():
                raise ValueError("T_model must return finite secondary states with the same leading dimensions as q.")
    with (out/"training.log").open("w") as log:
        with contextlib.ExitStack() as stack:
            if not verbose:
                stack.enter_context(contextlib.redirect_stdout(log))
                stack.enter_context(contextlib.redirect_stderr(log))
            trained = run_training(args, T_model=T_model, map_model=map_model,
                                   primal_norm_path=primal_norm_path, sec_norm_path=sec_norm_path,
                                   warm_start_path=warm_start_path, metric_primal=metric_primal,
                                   metric_secondary=metric_secondary)
    if not (out/"checkpoints/path_param.pth").exists():
        raise RuntimeError(f"Training did not finish; see {out/'training.log'}")
    metadata = {"version": 1, "config": cfg.to_dict(), "times": data.times.tolist(), "labels": data.labels,
                "dimension": data.primary[0].shape[1], "primary_scale": primary_scale,
                "secondary_scale": secondary_scale, "custom_map": not builtin_map and (T_model is not None or map_model is not None),
                "map_checkpoint": "map/model.pt" if builtin_map else None,
                "map_report": "map/report.json" if builtin_map else None,
                "blur_primary": args.blur_per_time_primary, "floor_primary": args.floor_pri_mean,
                "kernel_sigma": args.kernel_sigma,
                "blur_secondary": getattr(args, "blur_per_time_secondary", None)}
    (out/"result.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if args.sigma_per_cell_primary is not None:
        np.savez(out/"kernel_bandwidths.npz", **{f"t{i}": x.cpu().numpy() for i,x in enumerate(args.sigma_per_cell_primary)})
    result = Result(trained["func"], data, metadata, out, T_model=T_model, map_model=map_model)
    return result
