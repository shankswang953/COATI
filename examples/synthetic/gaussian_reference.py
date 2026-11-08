"""Self-contained reproduction of the original GaussToy BalancedSync experiment.

The observations and frozen-map weights are copied reference artifacts, not
new Gaussian samples. Numeric settings remain outside tutorial notebooks.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from coati import Config
from coati import TemporalData
from .gaussian_map import FourierBumpMLP


def _asset(name):
    return Path(__file__).resolve().parent.joinpath('data/gaussian_reference', name)



def load_example():
    """Load the exact original snapshots, frozen learned T and training settings."""
    settings = json.loads(_asset('settings.json').read_text())
    with _asset('primary.npz').open('rb') as stream, np.load(stream, allow_pickle=False) as data:
        primary = [data[key].copy() for key in ('time0', 'time1')]
    with _asset('secondary.npz').open('rb') as stream, np.load(stream, allow_pickle=False) as data:
        secondary = [data[key].copy() for key in ('time0', 'time1')]
    with _asset('map.pt').open('rb') as stream:
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
    config = dict(checkpoint['config']); config.pop('class_name', None)
    mapping = FourierBumpMLP(**config)
    mapping.load_state_dict(checkpoint['state_dict']); mapping.eval(); mapping.requires_grad_(False)
    dt = settings['prediction']['dt']
    times = torch.arange(0., 1. + .5*dt, dt, dtype=torch.float32).numpy()
    return SimpleNamespace(data=TemporalData(primary, settings['times'], secondary, ['time0', 'time1']),
                           mapping=mapping, config=Config(extra=settings['training']),
                           settings=settings, times=times)


def replay_reference():
    """Reintegrate the original velocity checkpoint with COATI's model class.

    Returns prediction arrays and their maximum absolute errors against the
    archived original trajectory tensors. This is checkpoint replay, not refitting.
    """
    from coati._core.Neural import MLPVectorField
    from coati.result import Result
    from pathlib import Path
    example = load_example(); cfg = example.config.to_dict()
    model = MLPVectorField(dim=2, hidden_dim=cfg['hidden_dim'], n_layers=cfg['n_layers'],
                           activation=cfg['activation'], unbalanced=False)
    with _asset('reference_velocity.pt').open('rb') as stream:
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['func_state_dict']); model.eval()
    metadata = {'config':cfg, 'dimension':2, 'primary_scale':1., 'secondary_scale':1.}
    result = Result(model, example.data, metadata, Path('.'), T_model=example.mapping)
    prediction = result.predict_trajectory(times=example.times, method=example.settings['prediction']['method'])
    time_grid = torch.tensor(prediction['times'], dtype=torch.float32)[:, None].expand(prediction['primary'].shape[:2])
    with torch.no_grad():
        prediction['secondary'] = example.mapping(torch.tensor(prediction['primary']), time_grid).numpy()
    errors = {}
    for kind in ('primary', 'secondary'):
        with _asset('reference_'+kind+'.pt').open('rb') as stream:
            saved = torch.load(stream, map_location='cpu', weights_only=True).numpy()
        errors[kind] = float(np.max(np.abs(prediction[kind]-saved)))
    return example, prediction, errors


def plot_reference(example, prediction):
    """Use the original figure's scales, observed points and seeded line selection."""
    import matplotlib.pyplot as plt
    time0, time1 = example.data.primary
    sec0, sec1 = example.data.secondary
    traj = torch.tensor(prediction['primary'])
    sec_traj = torch.tensor(prediction['secondary'])
    cfg = example.settings['prediction']
    line_idx = np.random.default_rng(cfg['seed']).choice(len(time0), min(cfg['n_lines'], len(time0)), replace=False)
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':10}):
        traj_np = traj.detach().cpu().numpy()
        sec_np = sec_traj.detach().cpu().numpy()
        terminal = traj_np[-1]
    
        fig = plt.figure(figsize=(13.8, 6.2), dpi=170, facecolor="white")
        ax2 = fig.add_subplot(1, 2, 1)
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    
        ax2.set_facecolor("white")
        ax2.scatter(time0[:, 0], time0[:, 1], s=10, c="#d62728", alpha=0.38, linewidths=0, label="time0")
        ax2.scatter(time1[:, 0], time1[:, 1], s=10, c="#ff7f0e", alpha=0.24, linewidths=0, label="time1")
        for idx in line_idx:
            path = traj_np[:, idx, :]
            ax2.plot(path[:, 0], path[:, 1], color="#315f8c", alpha=0.25, lw=0.75)
        ax2.scatter(terminal[:, 0], terminal[:, 1], s=12, c="#1b9e77", alpha=0.55, linewidths=0, label="pred")
        ax2.set_title("Primary 2D Trajectory")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_aspect("equal", adjustable="box")
        ax2.legend(frameon=True, facecolor="white", loc="lower right")
    
        for idx in line_idx:
            path = sec_np[:, idx, :]
            ax3.plot(path[:, 0], path[:, 1], path[:, 2], color="#315f8c", alpha=0.22, lw=0.75)
        ax3.scatter(sec0[:, 0], sec0[:, 1], sec0[:, 2], s=8, c="#1f77b4", alpha=0.25, linewidths=0, label="time0 sec")
        ax3.scatter(sec1[:, 0], sec1[:, 1], sec1[:, 2], s=8, c="#ff7f0e", alpha=0.20, linewidths=0, label="time1 sec")
        ax3.scatter(
            sec_np[-1, :, 0],
            sec_np[-1, :, 1],
            sec_np[-1, :, 2],
            s=10,
            c="#1b9e77",
            alpha=0.50,
            linewidths=0,
            label="pred sec",
        )
        ax3.set_title("Secondary 3D via Learned T")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_xlim(0.0, 1.0)
        ax3.set_ylim(0.0, 1.0)
        ax3.set_zlim(0.0, 1.05)
        ax3.view_init(elev=24, azim=-58)
        ax3.legend(frameon=True, facecolor="white", loc="upper left")
    
        fig.tight_layout()
        return fig
