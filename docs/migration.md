# Migrating from TraInf

COATI is an independent package. It does not import from, write to or require a local TraInf checkout. The original files were copied from the current working tree, including existing research changes, rather than assuming the committed public mirror represented the current algorithm.

| Previous workflow | COATI workflow |
| --- | --- |
| One working directory per experiment | One reusable package; separate output directories |
| `Args.py` parses process arguments | `Config` works in scripts and notebooks |
| NPZ paths embedded in configuration | `TemporalData` adapters and explicit time order |
| `sys.path` manipulation | Installable `coati` import |
| Checkpoint files alone | `Result`, input snapshots, configuration and metadata |
| Plot script tied to a dataset | `result.plot()` for the first two coordinates |

## Preserved engine

The neural field, loss assembly, optimizer, scheduler, adaptive blur, kernel projections, mass loss, dual updates and training discretization remain in `coati._core`. The full engine modules are retained for advanced use. The reusable `downstreamSrc` analysis modules are copied unchanged into `coati.downstream`; run `python -m coati.downstream.cli --help` or `coati-downstream --help`. Biological analyses are not executed in this validation. Flow matching, legacy plotting and biological helpers remain low-level modules; this release's high-level tutorials exercise dynamic OT and synchronized OT. Optional legacy plotting imports may require the `legacy` extra.

Only these mechanical changes were made inside the copied core:

1. `from src.*` becomes package-relative imports.
2. Remove the unused `from Args import get_args` import in `Epoch.py`.
3. Allow an explicit `args.device`; `device="auto"` follows the original selection.
4. Return zero gradient in the 100-step diagnostic for disabled scalar/non-differentiable losses. The original crashes in that diagnostic when synchronized density is disabled; training loss and optimizer updates are unchanged.

No loss constants, normalization formulas, neural architectures or update rules were changed. `evaluation.py` retains its secondary evaluation behavior. The wrapper checks inputs, stages numeric NPZ files, creates output directories, protects existing results and adds a reloadable result interface.

## Reproducing a previous experiment

Transfer all numerical settings, use the same data, row order, numeric times, normalization dictionaries, model/map, seed and device. Public defaults intentionally target small CPU examples; replacing a production `Args.py` with the default `Config()` is not equivalent.

Advanced engine arguments can be provided through `Config(extra={...})`. Data paths, times, dimensions and output locations are owned by the public adapter and derived from its inputs. Direct engine entry remains available as `coati._core.Training.run_training` for unsupported experimental features such as support-point workflows.

## Provenance

`provenance/core_sha256.json` records the original core file hashes. `scripts/verify_equivalence.py` compares original and packaged training with the same configuration and CPU seed, using read-only original imports with bytecode writing disabled. Private notes, personal configuration files, datasets and repository history are not copied.
