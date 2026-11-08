# Shared Gaussian-bump assets

These files are shared by the balanced and unbalanced synthetic tutorials.

- `observed.h5ad`: 1000 initial cells and 2000 terminal cells in paired primary
  and secondary representations.
- `map_pairs.h5ad`: observed pairs plus generated support used to fit T.
- `T.pt`: frozen FourierBumpMLP state and architecture metadata.
- `map_report.json`: fitting settings, held-out metrics and data parameters.

The observed count ratio is 2. The unbalanced tutorial also sets
`biological_num=[1.0, 2.0]` explicitly, so its target relative mass remains
well defined if the stored observations are later subsampled.

Regenerate the assets from repository code with:

```bash
python -m examples.synthetic.build_gaussian_assets \
  --output /path/to/new/gaussian_bump
```
