# Palate

This folder contains full-data and strict leave-one-stage-out COATI training for
the mouse secondary-palate data. The compact AnnData stores RNA PCA40 and ATAC
LSI15. Every configuration uses 20,000 iterations, batch size 1024, learning rate
`1e-4`, `dt=0.1`, seed 0 and `Cy=0.5`.

## Full-data training

```bash
python -m examples.palate.train_balanced \
  --output outputs/palate-balanced

python -m examples.palate.train_unbalanced \
  --output outputs/palate-unbalanced
```

Both commands use `data/T_FiLM_lsi15.pt`. Balanced uses width 400 with two
hidden layers. Unbalanced uses width 800 with three hidden layers, equal
population priors, `alpha_growth=1`, mass coefficient 1, `global_mass=False`
and adaptive primary/secondary ranges 5–50 and 10–100.

## Strict LOSO

The saved normalization scales are applied consistently across stages. The
held-out stage is excluded from vector-field training, and the selected strict
FiLM checkpoint was itself fitted without that stage.

Hold out E13.5 (LOO1):

```bash
python -m examples.palate.train_balanced_loso \
  --held-out-stage E13.5 \
  --output outputs/palate-balanced-loo-e13.5
```

Unbalanced, holding out E13.5:

```bash
python -m examples.palate.train_unbalanced_loso \
  --held-out-stage E13.5 \
  --output outputs/palate-unbalanced-loo-e13.5
```

Use `--held-out-stage E14.0` and a new output directory for LOO2. LOO1 uses
`T_FiLM_lsi15_LOTO_holdout_1.pt`; LOO2 uses
`T_FiLM_lsi15_LOTO_holdout_2.pt`. Balanced and unbalanced use the same settings
as the full-data commands, with the unbalanced equal-mass prior shortened to
the three retained stages.

Use `--iterations 2 --device cpu` only for a smoke test.
