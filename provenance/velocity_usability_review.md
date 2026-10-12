# Vector-field usability review

Baseline: `b5641d2`. No files under `src/coati/_core/` changed. Original TraInf repositories were not edited.

- Added explicit train_vector_field and modality/mass-conservation controls; old fit calls retain their behavior.
- Added readable parameter aliases and copied-config overrides. Explicit mode selection uses documented example presets; the selected config is saved with the original engine keys.
- Added velocity evaluation without integration, with normalization-scale reversal and per-cell numeric times.
- Added read-only AnnData visualization, explicit write_velocity storage, and array inputs. Neighbor-based local linear projection connects model coordinates to existing 2D embeddings. Masked grid smoothing supports arrows, grid and stream plots.
- Added CLI commands, a runnable toy script, an executed cell-by-cell notebook, and documentation.

38 tests passed, including exact parameter/trajectory equality for all four mode combinations. Analytic field and affine projection tests check units and direction. CLI train/plot and notebook execution passed. Two 100-iteration toy demos generated AnnData and figures. These checks establish API functionality, not biological accuracy or convergence; only synthetic CPU runs were used.

The embedding projection is an approximation, not an analytic UMAP Jacobian or scVelo's transition-probability estimator. Full-stage plots evaluate cells at their own times; use a subset or common time for a more specific view. Arrow lengths are display-scaled; stored velocities retain numerical magnitudes. See velocity_usability.json and docs/velocity.md.
