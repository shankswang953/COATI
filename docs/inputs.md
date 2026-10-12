# Input formats

All constructors produce `TemporalData`. Matrices contain cells in rows and prepared features in columns. Arrays are copied into finite `float32` matrices. No PCA, log transform, clipping or normalization is performed implicitly.

| Format | Constructor | Time information |
| --- | --- | --- |
| NumPy arrays / Torch tensors / sparse matrices | `TemporalData(snapshots, times)` | Explicit numeric times |
| Ordered mapping | `TemporalData.from_dict(mapping, times=...)` | Insertion order + explicit times |
| NPZ | `TemporalData.from_npz(path, keys=..., times=...)` | Explicit keys recommended |
| CSV / DataFrame | `from_csv` / `from_dataframe` | `time_key`, explicit feature columns |
| AnnData / h5ad | `from_anndata` / `from_h5ad` | `obs[time_key]`, optional `obsm` embeddings |

## Recommended AnnData fields

Prepare inputs once, before training or visualization. These are **COATI conventions
compatible with Scanpy**, not mandatory Scanpy column names.

| Location | Recommended content |
| --- | --- |
| `obs["celltype"]` | Supplied categorical cell-type labels |
| `obs["time"]` | Numeric times in model training units |
| `obs["barcode"]`, `obs_names` | The same globally unique cell identifiers |
| `obsm["X_umap"]` | Two coordinates for the same cells in the same order |

```python
cells = ct.prepare_anndata(adata, use_rep="X_pca",
                           time_key="day", celltype_key="cell_type", barcode_key="cell_id")
result = ct.train_vector_field(cells, balanced=False)
result.plot_dynamics(cells)
```

`prepare_anndata` returns a copy, keeps the original columns and row order, and
issues one notice describing adapted aliases or generated metadata. Canonical
inputs need no conversion notice. It recognizes common aliases such as
`cell_type`, `stage`, `time_point_processed` and `cell_id`; multiple candidate
columns require an explicit `*_key`. Categorical stages require `time_mapping`,
for example `{"E14.5": 0., "E15.5": 1., "E16.5": 2.}`. Times are never inferred
from the feature values or silently renumbered.

Missing labels are marked `unknown`. Array row identifiers can be generated,
while real barcodes must be unique. If barcodes repeat across batches/stages,
create explicit combined identifiers before conversion; the helper does not
silently deduplicate biological identities. If an existing index is replaced by
selected barcodes, it is retained in `obs["coati_original_obs_name"]`.

### Assemble arrays as AnnData

```python
cells = ct.prepare_anndata(embedding, time=cell_times, celltype=cell_types,
                           barcode=cell_ids, umap=umap_coordinates)
# Or use snapshots and explicitly request a display UMAP:
cells = ct.prepare_anndata([early, late], time=[0., 1.],
                           celltype=[early_types, late_types], compute_umap=True)
# TemporalData supplies its times and any paired secondary arrays:
cells = temporal_data.to_anndata(celltype=cell_types, compute_umap=True)
```

Prepared array features are recorded as `obsm["X_coati"]` and are not assumed to
be raw counts or PCA scores. `use_rep="auto"` reads the representation recorded
by preparation; for unprepared AnnData it looks for `X_pca`, then `X_latent`,
then `X_coati`, and finally X. Select `use_rep` explicitly when several plausible
representations are present. No normalization, count preprocessing or PCA is
performed by this helper. Existing UMAPs are reused; `compute_umap=True` explicitly
fits one on the selected prepared representation and refuses to replace an
existing UMAP.

For direct multimodal AnnData training, select paired features, for example
`ct.train_vector_field(cells, multimodal=True, secondary_obsm_key="X_atac")`.
All secondary rows must already correspond to the primary cells. `X_secondary`
from TemporalData conversion is recorded and selected automatically when the
multimodal mode is requested. Direct array training first goes through
`prepare_anndata`, so its times and identities are explicit.

## Arrays or mappings

```python
import coati as ct
import numpy as np

snapshots = [np.random.default_rng(i).normal(size=(100, 2)) for i in range(3)]
data = ct.TemporalData(snapshots, times=[0, .5, 1])
data = ct.TemporalData.from_dict(dict(early=snapshots[0], middle=snapshots[1], late=snapshots[2]), times=[0, .5, 1])
```

## Paired NPZ

```python
data = ct.TemporalData.from_npz(
    "rna.npz", secondary_path="atac.npz",
    keys=["early", "middle", "late"], times=[0, .5, 1],
)
```

Primary and secondary dimensions may differ. Within each time, rows must refer to the same cells in the same order. Row counts are checked; bare arrays cannot prove cell identity. Align identifiers before constructing arrays. Cross-time pairing is not required. Unpaired cross-modal datasets are not supported by this engine.

## CSV / DataFrame

```python
data = ct.TemporalData.from_csv(
    "embeddings.csv", time_key="stage", feature_keys=["PC1", "PC2"],
    secondary_keys=["LSI1", "LSI2", "LSI3"],
    time_order=["early", "middle", "late"], times=[0, .5, 1],
)
```

Categorical labels need an explicit numeric time mapping. `time_order` must include every observed label exactly once. Feature columns must be explicitly selected to avoid accidentally training on IDs or metadata.

## AnnData / h5ad

```python
data = ct.TemporalData.from_anndata(
    adata, time_key="stage", obsm_key="X_pca", secondary_obsm_key="X_lsi",
    time_order=["early", "middle", "late"], times=[0, .5, 1],
)
```

Both embeddings share `adata.obs`, preserving cell identity. Omit `obsm_key` to use `adata.X` explicitly. Sparse matrices are materialized densely; select a compact representation before importing large data. COATI does not claim automatic preprocessing of raw counts.

## Optional scalar normalization

The [biological reconstruction tutorial](tutorials/06_heldout_biological.ipynb)
includes an explicit `normalize` switch. Leave it off for prepared `X_latent`
inputs. Turn it on to estimate a W2-based scalar from the unscaled reduced
features in `X_reduced_raw`, independently for RNA and ATAC.

The scales are fitted using retained training stages only and then applied to
all observations and held-out references. They preserve relative feature scales;
this is not per-feature standardization or raw-count preprocessing. The notebook
saves the scales, the stages used to fit them and predictions converted back to
the selected input units. Existing PCA/PoissonVI representations remain fixed.
