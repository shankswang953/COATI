"""Load the current pretrained gastrulation models used by the NMP example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch

from coati.data import TemporalData
from coati.result import Result
from coati._core.Neural import MLPVectorField

from .film import load_film_checkpoint


ASSET_URL = (
    "https://raw.githubusercontent.com/shankswang953/COATI/main/"
    "examples/gastrulation/data/pretrained/nmp"
)
LOCAL_ASSETS = Path(__file__).resolve().parent / "data/pretrained/nmp"
DATA_PATH = Path(__file__).resolve().parent / "data/gastrulation_embeddings.h5ad"
MAP_PATH = Path(__file__).resolve().parent / "data/T_FiLM.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain_nmp_models(directory=None, *, base_url=ASSET_URL):
    """Return verified pretrained model paths, downloading missing weights."""
    directory = LOCAL_ASSETS if directory is None and LOCAL_ASSETS.is_dir() else Path(
        directory or "data/gastrulation/nmp"
    )
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        urllib.request.urlretrieve(f"{base_url}/manifest.json", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    paths = {}
    for name, record in manifest["assets"].items():
        path = directory / name
        if not path.exists():
            urllib.request.urlretrieve(f"{base_url}/{name}", path)
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")
        paths[name] = path
    return paths, manifest


def load_nmp_results(cells, *, device="cpu", model_directory=None):
    """Load balanced and unbalanced pretrained models without refitting."""
    paths, manifest = obtain_nmp_models(model_directory)
    data = TemporalData.from_anndata(
        cells,
        time_key="time",
        obsm_key="X_pca",
        secondary_obsm_key="X_atac",
        time_order=manifest["times"],
        times=manifest["times"],
    )
    mapping = load_film_checkpoint(MAP_PATH, device=device)
    results = {}
    for name, path in paths.items():
        spec = manifest["assets"][name]
        state = torch.load(path, map_location=device, weights_only=True)
        if int(state["iteration"]) != int(spec["iteration"]):
            raise ValueError(f"Iteration mismatch for {path}")
        model = MLPVectorField(
            dim=50,
            hidden_dim=int(spec["hidden_dim"]),
            n_layers=int(spec["n_layers"]),
            activation="leaky_relu",
            unbalanced=bool(spec["unbalanced"]),
            alpha_growth=float(spec["alpha_growth"]),
        ).to(device)
        model.load_state_dict(state["func_state_dict"])
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
            "times": manifest["times"],
            "labels": manifest["labels"],
            "dimension": 50,
            "primary_scale": float(manifest["primary_scale"]),
            "secondary_scale": float(manifest["secondary_scale"]),
            "custom_map": True,
            "map_checkpoint": None,
            "map_report": None,
            "checkpoint_iteration": int(spec["iteration"]),
            "cy": float(spec["cy"]),
            "checkpoint": name,
        }
        key = "unbalanced" if spec["unbalanced"] else "balanced"
        results[key] = Result(model, data, metadata, path.parent, T_model=mapping)
    return results, manifest


def validate_nmp_inputs(data_path=DATA_PATH, map_path=MAP_PATH):
    """Verify that local data and map files match the pretrained manifest."""
    manifest = json.loads((LOCAL_ASSETS / "manifest.json").read_text())
    checks = {
        "data": _sha256(Path(data_path)) == manifest["data_sha256"],
        "map": _sha256(Path(map_path)) == manifest["map_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"NMP input checksum mismatch: {checks}")
    return checks
