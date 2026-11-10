# Pancreas (moscot data)

This folder contains full-data and strict leave-one-stage-out COATI training for
the pancreas multiome data used in the moscot comparison. The compact AnnData
stores RNA PCA50 and ATAC PoissonVI22. All four configurations use 20,000
iterations, batch size 1024, learning rate `1e-4`, `dt=0.1`, seed 0 and `Cy=0.5`.

## Full-data training

```bash
python -m examples.moscot.train_balanced \
  --output outputs/moscot-balanced

python -m examples.moscot.train_unbalanced \
  --output outputs/moscot-unbalanced
```

Both commands use `data/T_FiLM_poissonvi.pt`. Balanced uses width 800, three
hidden layers and energy/density/PDF coefficients `0.1/0.1/100`. Unbalanced
uses width 800, three hidden layers, coefficients `0.01/0.01/100`, equal
population priors, `alpha_growth=1`, mass coefficient 1 and `global_mass=True`.
Both use adaptive lambda with primary and secondary bounds 10–100.

## Strict LOSO

E15.5 is excluded from vector-field training. The strict map
`T_FiLM_poissonvi_LOTO_holdout_1.pt` was also trained without E15.5.

Balanced:

```bash
python -m examples.moscot.train_balanced_loso \
  --output outputs/moscot-balanced-loo-e15.5
```

Unbalanced:

```bash
python -m examples.moscot.train_unbalanced_loso \
  --output outputs/moscot-unbalanced-loo-e15.5
```

The retained times are 0 and 2. The unbalanced equal-mass prior is `[1, 1]`.
Only the requested `alpha_growth=1` configuration is included; the incomplete
5 and 10 sweeps are not exposed here. Use `--iterations 2 --device cpu` only
for a smoke test.
