# Python API

```{eval-rst}
.. autofunction:: coati.anndata_io.prepare_anndata

.. autoclass:: coati.data.TemporalData
   :members:

.. autoclass:: coati.config.Config
   :members:

.. autoclass:: coati.config.MapConfig
   :members:

.. autofunction:: coati.training.train_vector_field

.. autofunction:: coati.training.fit

.. autoclass:: coati.result.Result
   :members:
```

## Toy datasets

```{eval-rst}
.. automodule:: coati.datasets
   :members:
```

## Projected velocity

```{eval-rst}
.. autoclass:: coati.velocity.VelocityEmbedding
   :members:

.. autofunction:: coati.velocity.project_velocity
```

## Paired map training

```{eval-rst}
.. autofunction:: coati.maps.train_map

.. autofunction:: coati.maps.load_map

.. autoclass:: coati.maps.PairedMap
   :members: forward
```

## Molecular trajectory readouts

```{eval-rst}
.. autofunction:: coati.downstream.find_gene_peaks

.. autofunction:: coati.downstream.compute_gene_peak_dynamics

.. autoclass:: coati.downstream.GenePeakDynamics
   :members: plot

.. autofunction:: coati.downstream.plot_gene_peak_dynamics

.. autofunction:: coati.downstream.read_h5ad_features

.. autofunction:: coati.downstream.fit_feature_decoder

.. autoclass:: coati.downstream.FeatureDecoder
   :members: predict, validate

.. autofunction:: coati.downstream.select_lineage

.. autofunction:: coati.downstream.compute_feature_dynamics

.. autofunction:: coati.downstream.plot_feature_dynamics

.. autofunction:: coati.downstream.plot_feature_heatmap
```

## Cell-type mass decomposition

```{eval-rst}
.. autofunction:: coati.downstream.decompose_mass

.. autoclass:: coati.downstream.MassDecomposition
   :members: plot

.. autofunction:: coati.downstream.plot_mass_decomposition
```
