# Adjusted Gaussian tutorial and original BalancedSync experiment

The adjusted tutorial is a separately constructed example. The reference version
loads the original NPZ observations and frozen learned map directly. It does not
regenerate a new sample from Gaussian parameters. All copied source artifacts
are verified against their original SHA-256 hashes.

| Setting | Adjusted tutorial | Original experiment |
| --- | --- | --- |
| Primary initial center | (0.30, 0.30) | (0.30, 0.30) |
| Terminal component centers | (0.56, 0.74), (0.74, 0.56) | (0.56, 0.82), (0.83, 0.56) |
| Observed sample counts | 512 initial, 512 terminal | 1,000 initial, 2,000 terminal |
| Data | Newly sampled tutorial fixture | Exact copied `rna_2d_by_time.npz` and `secondary_3d_bump_by_time.npz` |
| Data-generation seed | 7 | Generator default 2026; the copied arrays, rather than the generator, define the reference |
| Training seed | 7 | 0 |
| Secondary observations | Analytic bump pushforward | Same x/y observations, z exactly zero |
| Map T | Analytic Gaussian bump | Frozen `FourierBumpMLP`, original `T_bump_mlp.pt` |
| Secondary energy weight Cy | 0.9 | 0.35 |
| Velocity network | hidden 64, 3 layers, tanh | hidden 256, 2 layers, LeakyReLU |
| Training | shared 3,000-step warm-up, then 2,000 steps per comparison | 3,000 steps directly from scratch |
| Learning rate | 0.003 for warm-up, 0.0005 for refinement | 0.001 |
| Training dt | 0.05 | 0.1 |
| Primary/secondary marginal penalty | fixed 300 | adaptive lambda, initial 50, range 5..100 |
| Median / floor sample sizes | 128 / 64 | 512 / 256 |
| Minibatch size | 128 | 256 |
| Primary energy / manifold coefficients | 1 / 0 | 1 / 0 |
| Plot integration | RK4, 101 points | RK4, 51 points, dt=0.02 |
| Paths drawn | all 512 | seeded selection of 180 from 1,000 |
| 3D view | elevation 38, azimuth -58 | elevation 24, azimuth -58 |

Both use the unit square, observed times 0 and 1, and the same Gaussian kernel
covariances before truncation:

```text
Sigma0 = [[0.0042, 0.0002], [0.0002, 0.0038]]
Sigma1 = [[0.0030, 0.0001], [0.0001, 0.0030]]
Sigma2 = [[0.0032,-0.0001],[-0.0001, 0.0032]]
```

The nominal bump is centered at (0.5, 0.5), with width 0.08 and height 1. The
original learned map approximates this geometry and includes a small learned
x/y residual; it is not the analytic map used by the adjusted tutorial.

The original 1,000-to-2,000 counts are observation sample counts. This is a
balanced probability-distribution experiment, not a doubling of modeled mass.
No extra normalization is applied in either example.

## Reproduction guarantees

- The packaged original NPZ files, support cloud, T checkpoint, vector-field
  checkpoint, saved trajectories and original PNG match the original file hashes.
- Replaying the original vector-field checkpoint through COATI reproduces the
  primary saved trajectory exactly in the checked environment; mapped secondary
  coordinates differ by at most about 1.2e-7.
- Refitting with the reference settings produces a new result. It has not recovered
  the historical weights bit-for-bit. The historical checkpoint does not record
  the full software environment or command line, and the exact cause has not been
  isolated. The notebook displays replay and refitting separately.
- The reference uses the original source/configuration and verified log settings.
  Timing prints are disabled, as in the saved training log. Current refitting
  results and all numerical differences are in `gaussian_reference_validation.json`.

The original TraInf files were read and copied, never modified.
