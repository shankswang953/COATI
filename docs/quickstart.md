# Quick start

```python
import coati as ct

result = ct.train_vector_field(ct.datasets.gaussian(), iterations=200)
trajectory = result.predict_trajectory()
result.plot_trajectory()
```

## Choose modalities and population dynamics

| `multimodal` | `balanced` | Model |
| --- | --- | --- |
| `False` | `True` | Single-modality transport; default |
| `True` | `True` | Paired multi-modal transport |
| `False` | `False` | Single-modality transport with growth/mass change |
| `True` | `False` | Paired multi-modal transport with growth/mass change |

```python
config = ct.Config(multimodal=True, balanced=False,
                   iterations=500, learning_rate=1e-3, batch_size=64,
                   secondary_weight=.35)
result = ct.train_vector_field(ct.datasets.paired(), config)
```

An explicit multimodal choice uses secondary_weight=.35 unless overridden.
Explicit unbalanced mode uses growth_weight=1, mass_loss=True and mass_weight=1.
These presets are example starting values; inspect `config.to_dict()` and provide
settings for your data. Paired runs require corresponding primary/secondary rows
within each snapshot. A run with secondary data can still select multimodal=False.
The automatic FiLM/MLP map is trained first and frozen, as before.

## Use your AnnData

```python
data = ct.TemporalData.from_anndata(
    adata, time_key="time", obsm_key="X_pca",
    secondary_obsm_key="X_atac",  # omit for a primary-only input
)
result = ct.train_vector_field(data, multimodal=True, balanced=True)
result.plot_velocity(adata, use_rep="X_pca", time_key="time",
                     color="cell_type", style="stream")
```

For a single-modality run omit secondary_obsm_key and use multimodal=False.
Use the same features and preprocessing for plotting as for training. Existing
UMAP coordinates must already be stored in adata.obsm["X_umap"]. Plotting supports
string stage labels with an explicit numeric `time_mapping`; see [velocity](velocity.md).

## Override or reuse settings

```python
config = ct.Config(iterations=500, learning_rate=1e-3)
result = ct.train_vector_field(data, config, iterations=1000, output_dir="outputs/run1")
config.save("config.json")
config = ct.Config.load("config.json")
restored = ct.load_result(result.output_dir)
velocity = restored.compute_velocity()  # evaluates at each observed cell's time
```

Overrides do not mutate the original Config. Output directories must be new.
Old fit/predict/plot names remain available for existing scripts.

## Scripts and notebooks

```bash
python examples/run_velocity.py --output outputs/velocity-single
python examples/run_velocity.py --multimodal --unbalanced --output outputs/velocity-multi
coati train-vector-field primary.npz --keys t0 t1 --times 0 1 --iterations 200 --output outputs/run
coati plot-velocity outputs/run cells.h5ad --use-rep X_pca --time-key time --color cell_type --output velocity.pdf
python -m jupyterlab notebooks/05_velocity.ipynb
```

Install the visualization extra for the AnnData/UMAP toy script. It exports an
AnnData with computed vectors and arrow/grid/stream figures. Existing embeddings
can be plotted without fitting UMAP. Short toy runs verify functionality; they do
not establish convergence or biological accuracy.
