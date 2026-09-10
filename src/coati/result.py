"""Reloadable model, trajectory prediction, metrics and lightweight plotting."""
from pathlib import Path
import json
import numpy as np
from .data import TemporalData

class Result:
    def __init__(self, model, data, metadata, output_dir, T_model=None, map_model=None):
        self.model = model.eval()
        self.data, self.metadata = data, metadata
        self.output_dir = Path(output_dir)
        self.T_model = T_model
        self.map_model = map_model

    @classmethod
    def load(cls, directory, *, device="cpu", T_model=None, map_model=None):
        """Reload weights and input snapshots. Custom map architecture is supplied by the caller."""
        import torch
        from ._core.Neural import MLPVectorField
        directory = Path(directory)
        meta = json.loads((directory/"result.json").read_text())
        cfg = meta["config"]
        model = MLPVectorField(dim=meta["dimension"], hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
                               activation=cfg["activation"], unbalanced=cfg["unbalancedModel"],
                               alpha_growth=cfg["alpha_growth"]).to(device)
        ckpt = torch.load(directory/"checkpoints/path_param.pth", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["func_state_dict"])
        secondary = directory/"inputs/secondary.npz"
        data = TemporalData.from_npz(directory/"inputs/primary.npz", times=meta["times"],
                                     secondary_path=secondary if secondary.exists() else None)
        import copy
        def freeze(m):
            if m is None: return None
            m=copy.deepcopy(m).to(device).eval()
            for p in m.parameters(): p.requires_grad_(False)
            return m
        return cls(model, data, meta, directory, freeze(T_model), freeze(map_model))

    def predict(self, initial=None, *, times=None, method="rk4", raw=True):
        """Return {times, primary, energy, log_weights?} on an increasing grid.

        initial is in input units when raw=True. Returned energy remains in training units.
        times starts at the observed initial time; its spacing controls fixed-step solvers.
        """
        import torch
        from torchdiffeq import odeint
        device = next(self.model.parameters()).device
        t = np.asarray(times if times is not None else np.linspace(self.data.times[0], self.data.times[-1], 41), dtype=float)
        if t.ndim != 1 or len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
            raise ValueError("Prediction times must be finite and strictly increasing.")
        if not np.isclose(t[0], self.data.times[0]):
            raise ValueError("Prediction must start at the initial observed time.")
        x = torch.as_tensor(self.data.primary[0] if initial is None else initial, dtype=torch.float32, device=device)
        if x.ndim != 2 or x.shape[1] != self.metadata["dimension"] or not torch.isfinite(x).all():
            raise ValueError("Initial states must be a finite cells × model-dimension matrix.")
        scale = self.metadata["primary_scale"]
        if raw or initial is None: x = x/scale
        zeros = x.new_zeros((len(x),1))
        grid = torch.as_tensor(t, dtype=x.dtype, device=device)
        with torch.no_grad():
            if self.metadata["config"]["unbalancedModel"]:
                state = odeint(self.model, (x, zeros-np.log(len(x)), zeros, zeros), grid, method=method)
                trajectory, logw, energy, growth = state
            else:
                trajectory, energy = odeint(self.model, (x,zeros), grid, method=method)
                logw = None
        answer = {"times": t, "primary": trajectory.cpu().numpy()*(scale if raw else 1),
                  "energy": energy.cpu().numpy()}
        if logw is not None:
            answer.update(log_weights=logw.cpu().numpy(), growth_energy=growth.cpu().numpy())
        return answer

    def project_secondary(self, prediction=None, *, raw=True, chunk_size=1024):
        """Project a primary trajectory using the trained map and return secondary coordinates.

        Pass a prediction in input units when raw=True (the default). Kernel prediction
        uses all paired observations as a deterministic basis; training retains its original
        randomly sampled basis. chunk_size limits query memory, not basis size.
        """
        import torch
        from ._core.MapSpace import kernel_project_paired
        if self.data.secondary is None or not self.metadata["config"]["sync_loss"]:
            raise ValueError("Secondary projection requires a synchronized paired run.")
        prediction = self.predict(raw=raw) if prediction is None else prediction
        device = next(self.model.parameters()).device
        q = torch.as_tensor(prediction["primary"], dtype=torch.float32, device=device)
        if raw: q = q/self.metadata["primary_scale"]
        grid = torch.as_tensor(prediction["times"],dtype=q.dtype,device=device)
        with torch.no_grad():
            if self.T_model is not None:
                mapped=self.T_model(q,grid[:,None].expand(q.shape[:2]))
            elif self.map_model is not None:
                mapped=self.map_model.map_rna_to_atac(q.reshape(-1,q.shape[-1])).reshape(*q.shape[:2],-1)
            elif self.metadata["custom_map"]:
                raise ValueError("Reload with the original T_model or map_model to project secondary trajectories.")
            else:
                x=torch.tensor(np.concatenate(self.data.primary)/self.metadata["primary_scale"],device=device)
                y=torch.tensor(np.concatenate(self.data.secondary)/self.metadata["secondary_scale"],device=device)
                sigma=self.metadata["kernel_sigma"]
                if self.metadata["config"]["map_type"] == "accurate_kernel":
                    with np.load(self.output_dir/"kernel_bandwidths.npz",allow_pickle=False) as f:
                        sigma=torch.tensor(np.concatenate([f[f"t{i}"] for i in range(len(self.data.times))]),device=device)
                if chunk_size < 1: raise ValueError("chunk_size must be positive.")
                flat=q.reshape(-1,q.shape[-1])
                mapped=torch.cat([kernel_project_paired(part,x,y,sigma) for part in flat.split(chunk_size)])
                mapped=mapped.reshape(*q.shape[:2],-1)
        return mapped.cpu().numpy()*(self.metadata["secondary_scale"] if raw else 1.)

    def evaluate(self):
        """Endpoint Sinkhorn before/after transport; descriptive toy sanity check, not a biological benchmark."""
        import torch
        from geomloss import SamplesLoss
        pred = self.predict(raw=False)
        x = torch.tensor(self.data.primary[0]/self.metadata["primary_scale"])
        y = torch.tensor(self.data.primary[-1]/self.metadata["primary_scale"])
        z = torch.tensor(pred["primary"][-1])
        loss = SamplesLoss("sinkhorn", p=2, blur=self.metadata["blur_primary"][-1], backend="tensorized")
        with torch.no_grad():
            scores = {"endpoint_sinkhorn_before": float(loss(x,y)), "endpoint_sinkhorn_after": float(loss(z,y)),
                      "kinetic_energy_mean": float(pred["energy"][-1].mean()),
                      "endpoint_sampling_floor": self.metadata["floor_primary"][-1]}
        if self.metadata["config"]["sync_loss"] and (not self.metadata["custom_map"] or self.T_model is not None or self.map_model is not None):
            secondary = self.project_secondary(pred, raw=False)
            target = torch.tensor(self.data.secondary[-1]/self.metadata["secondary_scale"])
            secondary_loss = SamplesLoss("sinkhorn",p=2,blur=self.metadata["blur_secondary"][-1],backend="tensorized")
            scores["secondary_endpoint_sinkhorn"] = float(secondary_loss(torch.tensor(secondary[-1]),target))
        if "log_weights" in pred:
            scores["predicted_mass_ratio"] = float(np.exp(pred["log_weights"][-1]).sum())
            scores["observed_count_ratio"] = len(y)/len(x)
        return scores

    def plot(self, *, ax=None, max_paths=30):
        """Plot the first two input coordinates; does not fit an embedding."""
        import matplotlib.pyplot as plt
        if self.metadata["dimension"] < 2: raise ValueError("plot requires at least two features.")
        if ax is None: _, ax = plt.subplots(figsize=(6.5,4.5))
        pred = self.predict()
        for i,(t,x) in enumerate(zip(self.data.times,self.data.primary)):
            ax.scatter(x[:,0],x[:,1],s=10,alpha=.35,label=f"Observed t={t:g}")
        for path in pred["primary"][:,:max_paths,:].transpose(1,0,2):
            ax.plot(path[:,0],path[:,1],color="#0072ce",alpha=.5,lw=1)
        ax.set(xlabel="Primary dimension 1",ylabel="Primary dimension 2",title="COATI · inferred trajectories")
        ax.legend(frameon=False)
        return ax
