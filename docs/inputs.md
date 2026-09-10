# Input formats

All constructors produce `TemporalData`. Matrices contain cells in rows and prepared features in columns. Arrays are copied into finite `float32` matrices. No PCA, log transform, clipping or normalization is performed implicitly.

| Format | Constructor | Time information |
| --- | --- | --- |
| NumPy arrays / Torch tensors / sparse matrices | `TemporalData(snapshots, times)` | Explicit numeric times |
| Ordered mapping | `TemporalData.from_dict(mapping, times=...)` | Insertion order + explicit times |
| NPZ | `TemporalData.from_npz(path, keys=..., times=...)` | Explicit keys recommended |
| CSV / DataFrame | `from_csv` / `from_dataframe` | `time_key`, explicit feature columns |
| AnnData / h5ad | `from_anndata` / `from_h5ad` | `obs[time_key]`, optional `obsm` embeddings |

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
