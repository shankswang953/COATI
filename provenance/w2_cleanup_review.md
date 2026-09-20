# W2-only cleanup review

Baseline: `e05b2f8c70e20426bebae3016b2453fbb000f7d1`.

- Removed the MMD class from `Loss.py`, primary/reverse/secondary MMD branches, result keys, TensorBoard entries, and default flags.
- Removed `RiemannMetric.py` and the `metric_primal` / `metric_secondary` arguments throughout the training call chain.
- Retained the original `p=2` Sinkhorn matching, blur, weighting, Euclidean energy expression, learned-map training, mass losses and evaluation diagnostics.
- Older disabled MMD flags are discarded on configuration loading. Enabled MMD flags raise an explicit error. Removed metric keyword arguments now raise TypeError.
- Original TraInf files were not modified. The original-engine comparison adapter supplies the old disabled flags only to the original engine.

## Validation

All 28 tests passed. A deliberately short FiLM toy fit emitted its expected map-quality warning. Five before/after CPU toy cases (four iterations each) produced exactly equal model parameters and predicted trajectories, and predictions reloaded identically. See `w2_cleanup.json` for per-case differences.

The baseline reverse-matching branch already fails with a state-shape IndexError in `Neural.py`; this cleanup does not repair that separate issue. Reverse matching is excluded from equivalence claims. No biological datasets or GPU runs were used.

## Reproduce the comparison

Extract `src/coati` from the baseline commit into a fresh directory without modifying a working checkout. With the same Python environment, run:

```bash
PYTHONPATH=/path/to/baseline/src python scripts/verify_matching_cleanup.py /tmp/baseline-results
PYTHONPATH=src python scripts/verify_matching_cleanup.py /tmp/current-results /tmp/baseline-results
```

Use fresh output directories. The comparison script excludes the known baseline reverse-matching failure.
