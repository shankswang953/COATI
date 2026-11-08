"""Load the archived full-data gastrulation checkpoints for inference."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import urllib.request

import torch

from coati.data import TemporalData
from coati.result import Result
from coati._core.Neural import MLPVectorField


ASSET_URL = (
    "https://raw.githubusercontent.com/shankswang953/COATI/main/"
    "examples/data/gastrulation/pretrained"
)
LOCAL_ASSETS = Path(__file__).resolve().parent / "data/gastrulation/pretrained"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain_gastrulation_assets(directory=None, *, base_url=ASSET_URL):
    """Return verified local paths, downloading public assets when necessary."""
    directory = LOCAL_ASSETS if directory is None and LOCAL_ASSETS.is_dir() else Path(
        directory or "data/gastrulation/pretrained"
    )
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        urllib.request.urlretrieve(f"{base_url}/manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    paths = {"manifest": manifest_path}
    for name, record in manifest["assets"].items():
        path = directory / name
        if not path.exists():
            urllib.request.urlretrieve(f"{base_url}/{name}", path)
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")
        paths[name] = path
    return paths, manifest


def load_gastrulation_result(checkpoint, primary_data, *, device="cpu", manifest=None):
    """Load one archived checkpoint as a COATI Result without refitting."""
    checkpoint = Path(checkpoint)
    primary_data = Path(primary_data)
    if manifest is None:
        manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    try:
        spec = manifest["assets"][checkpoint.name]
    except KeyError as exc:
        raise ValueError(f"Checkpoint is not described by the gastrulation manifest: {checkpoint.name}") from exc
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if int(state["iteration"]) != int(spec["iteration"]):
        raise ValueError("Checkpoint iteration does not match the manifest.")
    model = MLPVectorField(
        dim=50,
        hidden_dim=int(spec["hidden_dim"]),
        n_layers=int(spec["n_layers"]),
        activation="leaky_relu",
        unbalanced=bool(spec["unbalanced"]),
        alpha_growth=float(spec["alpha_growth"]),
    ).to(device)
    model.load_state_dict(state["func_state_dict"])
    data = TemporalData.from_npz(
        primary_data,
        times=manifest["times"],
        keys=manifest["labels"],
    )
    config = {
        "hidden_dim": int(spec["hidden_dim"]),
        "n_layers": int(spec["n_layers"]),
        "activation": "leaky_relu",
        "unbalancedModel": bool(spec["unbalanced"]),
        "alpha_growth": float(spec["alpha_growth"]),
        "sync_loss": True,
    }
    metadata = {
        "version": 1,
        "config": config,
        "times": list(manifest["times"]),
        "labels": list(manifest["labels"]),
        "dimension": 50,
        "primary_scale": float(manifest["primary_scale"]),
        "secondary_scale": 1.0,
        "custom_map": True,
        "map_checkpoint": None,
        "map_report": None,
        "legacy_checkpoint": checkpoint.name,
        "checkpoint_iteration": int(spec["iteration"]),
        "cy": float(spec["cy"]),
    }
    return Result(model, data, metadata, checkpoint.parent)


def load_gastrulation_umap(path, primary_data=None):
    """Load the original RNA UMAP reducer, including older pynndescent pickles."""
    import joblib
    import numpy as np
    import pynndescent

    original_setstate = pynndescent.NNDescent.__setstate__

    def compatible_setstate(index, state):
        state.setdefault("quantization", None)
        state.setdefault("_min_distance", 0.0)
        state.setdefault("parallel_batch_queries", False)
        return original_setstate(index, state)

    pynndescent.NNDescent.__setstate__ = compatible_setstate
    try:
        reducer = joblib.load(path)
    finally:
        pynndescent.NNDescent.__setstate__ = original_setstate
    if primary_data is not None:
        with np.load(primary_data, allow_pickle=False) as archive:
            observed = np.concatenate([archive[key] for key in ("time0", "time1", "time2", "time3")])
        if reducer._raw_data.shape != observed.shape or not np.array_equal(reducer._raw_data, observed):
            raise ValueError("The UMAP reducer was not fitted on the supplied RNA snapshots in their stored order.")
    return reducer
