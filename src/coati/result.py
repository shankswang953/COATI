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
        report_path = metadata.get("map_report")
        self.map_report = json.loads((self.output_dir / report_path).read_text()) if report_path else None

    @classmethod
    def load(cls, directory, *, device="cpu", T_model=None, map_model=None):
        """Reload trajectories and built-in maps; external maps must be supplied again."""
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
        from .maps import load_map, freeze_external_map
        if T_model is not None and map_model is not None:
            raise ValueError("Supply only one of T_model and map_model.")
        if T_model is None and map_model is None and meta.get("map_checkpoint"):
            T_model = load_map(directory / meta["map_checkpoint"], device=device)
        elif T_model is not None:
            T_model = freeze_external_map(T_model, device)
        import copy
        def freeze(m):
            if m is None: return None
            m=copy.deepcopy(m).to(device).eval()
            for p in m.parameters(): p.requires_grad_(False)
            return m
        return cls(model, data, meta, directory, T_model, freeze(map_model))

    def predict(self, initial=None, *, times=None, method="rk4", raw=True, start_time=None):
        """Return {times, primary, energy, log_weights?} on an increasing grid.

        initial is in input units when raw=True. Returned energy remains in training units.
        times starts at the observed initial time by default; its spacing controls fixed-step solvers.
        start_time allows an explicit supplied cohort to start later within the observed range.
        It does not select, transport or relabel initial cells automatically.
        """
        import torch
        from torchdiffeq import odeint
        device = next(self.model.parameters()).device
        start = float(self.data.times[0] if start_time is None else start_time)
        if not np.isfinite(start):
            raise ValueError("start_time must be finite.")
        if start_time is not None:
            if not self.data.times[0] <= start < self.data.times[-1]:
                raise ValueError("start_time must lie within the observed range, before the final time.")
            if initial is None and not np.isclose(start, self.data.times[0]):
                raise ValueError("Supply initial states when starting at a later time.")
        t = np.asarray(times if times is not None else np.linspace(start, self.data.times[-1], 41), dtype=float)
        if t.ndim != 1 or len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
            raise ValueError("Prediction times must be finite and strictly increasing.")
        if not np.isclose(t[0], start):
            raise ValueError("Prediction times must begin at start_time (the initial observed time by default).")
        if start_time is not None and t[-1] > self.data.times[-1] + 1e-8:
            raise ValueError("A later-start cohort must end within the observed range.")
        x = torch.as_tensor(self.data.primary[0] if initial is None else initial, dtype=torch.float32, device=device)
        if x.ndim != 2 or not len(x) or x.shape[1] != self.metadata["dimension"] or not torch.isfinite(x).all():
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


    def predict_trajectory(self, initial=None, **kwargs):
        """Integrate the trained vector field; explicit name for predict()."""
        return self.predict(initial, **kwargs)

    def plot_trajectory(self, **kwargs):
        """Plot integrated paths; explicit name for plot()."""
        return self.plot(**kwargs)

    def compute_velocity(self, points=None, *, time=None, raw=True, chunk_size=1024):
        """Evaluate the learned field without integrating trajectories.

        points must use the same features, order and preprocessing as training.
        time is a scalar or one numeric time per cell. With points=None, snapshot
        rows are concatenated in data order and use their observed times. Explicit
        points require explicit times. raw=True returns input-unit displacement
        per unit of the numeric time supplied to training.
        """
        import torch
        from .velocity import _matrix
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if points is None:
            points = np.concatenate(self.data.primary)
            if not raw:
                points = points / self.metadata["primary_scale"]
            if time is None:
                time = np.repeat(self.data.times, [len(x) for x in self.data.primary])
        elif time is None:
            raise ValueError("Provide a numeric time when evaluating explicit points.")
        points = _matrix(points, "points", self.metadata["dimension"])
        try:
            times = np.broadcast_to(np.asarray(time, dtype=float), (len(points),))
        except ValueError as exc:
            raise ValueError("time must be a scalar or one numeric value per cell.") from exc
        if not np.isfinite(times).all():
            raise ValueError("time must contain finite numeric values.")
        parameter = next(self.model.parameters())
        scale = self.metadata["primary_scale"] if raw else 1.
        output = []
        with torch.no_grad():
            for start in range(0, len(points), chunk_size):
                x = torch.tensor(points[start:start+chunk_size] / scale, dtype=parameter.dtype, device=parameter.device)
                t = torch.tensor(times[start:start+chunk_size, None].copy(), dtype=x.dtype, device=x.device)
                output.append(self.model.vector_field_net(torch.cat([x, t], dim=1)).cpu().numpy() * scale)
        return np.concatenate(output)

    def project_velocity(self, embedding, *, points=None, time=None, raw=True,
                         n_neighbors=30, ridge=1e-3, basis="umap"):
        """Project corresponding representation vectors into an existing 2D embedding."""
        from .velocity import project_velocity
        velocity = self.compute_velocity(points, time=time, raw=raw)
        if points is None:
            points = np.concatenate(self.data.primary)
            if not raw:
                points = points / self.metadata["primary_scale"]
            if time is None:
                time = np.repeat(self.data.times, [len(x) for x in self.data.primary])
        return project_velocity(points, velocity, embedding, time=time, basis=basis,
                                n_neighbors=n_neighbors, ridge=ridge)

    def compute_velocity_embedding(self, adata, *, use_rep="auto", time_key="time",
                                   basis="umap", time=None, time_mapping=None,
                                   n_neighbors=30, ridge=1e-3):
        """Return projected vectors for AnnData rows, without modifying AnnData.

        use_rep=None selects adata.X. Numeric observation times must match training
        units; categorical stages require an explicit label-to-time mapping.
        """
        from .velocity import from_anndata
        return from_anndata(self, adata, use_rep=use_rep, time_key=time_key, basis=basis,
                            time=time, time_mapping=time_mapping, n_neighbors=n_neighbors, ridge=ridge)

    def write_velocity(self, adata, *, key="coati_velocity", copy=False, **kwargs):
        """Explicitly store primary and embedding velocities in AnnData.obsm.

        Returns the input AnnData or a copy. Reordered cells retain their own
        representation, times and embedding; training row order is not assumed.
        """
        from .velocity import _matrix
        target = adata.copy() if copy else adata
        field = self.compute_velocity_embedding(target, **kwargs)
        from .anndata_io import resolve_representation
        use_rep = resolve_representation(target, kwargs.get("use_rep", "auto"))
        points = _matrix(target.X if use_rep is None else target.obsm[use_rep], "training representation")
        target.obsm[key] = self.compute_velocity(points, time=field.time)
        target.obsm[f"{key}_{field.basis}"] = field.velocity
        target.uns[key] = dict(basis=field.basis, use_rep="X" if use_rep is None else use_rep,
                               time_key=kwargs.get("time_key", "time"), projection=field.projection,
                               source="COATI learned primary-space vector field")
        return target

    def plot_velocity(self, adata=None, *, embedding=None, points=None, time=None,
                      use_rep="auto", time_key="time", time_mapping=None, basis="umap",
                      n_neighbors=30, ridge=1e-3, color=None, **plot_options):
        """Draw streamlines, grid arrows or cell arrows on an existing embedding.

        AnnData example: result.plot_velocity(adata, use_rep="auto", time_key="time",
        color="cell_type", style="stream"). NumPy inputs instead use embedding,
        points and time. Plotting never changes AnnData or fits an embedding.
        """
        if adata is not None:
            if embedding is not None or points is not None:
                raise ValueError("Use either AnnData or explicit embedding/points arrays.")
            field = self.compute_velocity_embedding(adata, use_rep=use_rep, time_key=time_key,
                       time_mapping=time_mapping, basis=basis, time=time, n_neighbors=n_neighbors, ridge=ridge)
            if isinstance(color, str) and color in adata.obs:
                color = adata.obs[color]
        else:
            if embedding is None:
                raise ValueError("Provide AnnData or an existing two-dimensional embedding.")
            field = self.project_velocity(embedding, points=points, time=time, basis=basis,
                                           n_neighbors=n_neighbors, ridge=ridge)
        return field.plot(color=color, n_neighbors=n_neighbors, **plot_options)

    def compute_growth(self, points=None, *, time=None, raw=True, chunk_size=1024):
        """Evaluate instantaneous d log(weight)/dt for an unbalanced model.

        The rate follows the saved engine's actual ODE: growth_net(x,t) times
        model.alpha_growth. raw controls input coordinate units, not rate units.
        Time units are exactly those supplied during training. This is distinct
        from cumulative log_weights or total population mass.
        """
        import torch
        from .velocity import _matrix
        if not self.metadata["config"]["unbalancedModel"] or not hasattr(self.model, "growth_net"):
            raise ValueError("Growth rates require an unbalanced model trained with balanced=False.")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if points is None:
            points = np.concatenate(self.data.primary)
            if not raw:
                points = points / self.metadata["primary_scale"]
            if time is None:
                time = np.repeat(self.data.times, [len(x) for x in self.data.primary])
        elif time is None:
            raise ValueError("Provide a numeric time when evaluating explicit points.")
        points = _matrix(points, "points", self.metadata["dimension"])
        try:
            times = np.broadcast_to(np.asarray(time, dtype=float), (len(points),))
        except ValueError as exc:
            raise ValueError("time must be a scalar or one numeric value per cell.") from exc
        if not np.isfinite(times).all():
            raise ValueError("time must contain finite numeric values.")
        parameter = next(self.model.parameters())
        scale = self.metadata["primary_scale"] if raw else 1.
        output = []
        with torch.no_grad():
            for start in range(0, len(points), chunk_size):
                x = torch.tensor(points[start:start+chunk_size] / scale, dtype=parameter.dtype, device=parameter.device)
                t = torch.tensor(times[start:start+chunk_size, None].copy(), dtype=x.dtype, device=x.device)
                rate = self.model.growth_net(torch.cat([x, t], dim=1)) * self.model.alpha_growth
                if rate.shape != (len(x), 1) or not torch.isfinite(rate).all():
                    raise ValueError("The growth model must return one finite scalar rate per cell.")
                output.append(rate[:, 0].cpu().numpy())
        return np.concatenate(output)

    def plot_growth(self, adata, *, use_rep="auto", time_key="time", time=None,
                    time_mapping=None, basis="umap", celltype_key="celltype", labels=True,
                    show_velocity=False, n_neighbors=30, ridge=1e-3, ax=None, **plot_options):
        """Plot instantaneous growth on UMAP, optionally with streams and labels.

        Set show_velocity=True to overlay COATI flow. Growth uses original model
        features; color is the actual d log(weight)/dt, not cumulative log weight.
        """
        from .dynamics import plot_growth
        return plot_growth(self, adata, use_rep=use_rep, time_key=time_key, time=time,
            time_mapping=time_mapping, basis=basis, celltype_key=celltype_key, labels=labels,
            show_velocity=show_velocity, n_neighbors=n_neighbors, ridge=ridge, ax=ax, **plot_options)

    def plot_dynamics(self, adata, *, use_rep="auto", time_key="time", time=None,
                      time_mapping=None, basis="umap", celltype_key="celltype", labels=True,
                      include_growth=None, palette=None, n_neighbors=30, ridge=1e-3,
                      axes=None, **plot_options):
        """Plot cell types and COATI streams; add growth for unbalanced models.

        Returns a flat array of Axes. Each cell uses its observation time unless
        time is supplied. Subset AnnData for a one-stage view. See coati.dynamics.
        """
        from .dynamics import plot_dynamics
        return plot_dynamics(self, adata, use_rep=use_rep, time_key=time_key, time=time,
            time_mapping=time_mapping, basis=basis, celltype_key=celltype_key, labels=labels,
            include_growth=include_growth, palette=palette, n_neighbors=n_neighbors, ridge=ridge,
            axes=axes, **plot_options)

    def write_dynamics(self, adata, *, copy=False, growth_key="coati_growth", **kwargs):
        """Store velocity vectors and, for unbalanced models, instantaneous growth.

        Returns AnnData. copy=True preserves the caller's input. Observation
        identities and row order are retained; no UMAP is fitted here.
        """
        from .anndata_io import resolve_representation, observation_times
        from .velocity import _matrix
        target = adata.copy() if copy else adata
        rep = resolve_representation(target, kwargs.get("use_rep", "auto"))
        times = observation_times(target, time=kwargs.get("time"), time_key=kwargs.get("time_key", "time"),
                                  time_mapping=kwargs.get("time_mapping"))
        target = self.write_velocity(target, **kwargs)
        if self.metadata["config"]["unbalancedModel"]:
            points = _matrix(target.X if rep is None else target.obsm[rep], "training representation")
            target.obs[growth_key] = self.compute_growth(points, time=times)
            target.uns[growth_key] = dict(definition="d log(weight) / dt", units="per model time unit",
                                         source="COATI saved unbalanced ODE")
        return target
