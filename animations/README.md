# COATI animated introduction

A silent, captioned Manim film for the documentation homepage. Blue denotes the primary/RNA space; orange denotes the secondary/ATAC space. All particle positions come from analytic illustration functions, not a fitted model or biological dataset.

## Reproduce

```bash
python -m venv .venv-manim
source .venv-manim/bin/activate
pip install -r animations/requirements.txt
python animations/render.py
```

Manim's native Cairo/Pango dependencies must be available for the platform. Formulas use Matplotlib mathtext exported as SVG, so this scene does not require LaTeX/dvisvgm. Use `python animations/render.py --preview` for 720p; the default is 1080p. The video is silent by design and includes on-screen explanations.

## Scientific scope

This is the balanced core, with a fixed (possibly time-dependent) map during trajectory fitting. It does not attempt to explain the growth/mass extension in the same introductory clip.

1. At each observed time, the two modalities are paired. Across-time trajectories are not observed.
2. Only the primary velocity is parameterized as a Neural ODE: `dx/dt = u_theta(x,t)`.
3. Secondary states are `y(t) = T_omega(x(t),t)`. The full chain rule includes both `D_x T u_theta` and `partial_t T`.
4. `xi_t = (T_{omega,t})_# rho_t` is the predicted pushforward distribution. It is not asserted equal to the empirical secondary snapshot by definition.
5. Both spaces constrain theta. T is frozen in this illustration; gradients pass through its input.
6. `c = sync_weight` mixes energy and manifold terms. It does not multiply or disable the independent Sinkhorn coefficients. Changing c in the film changes a coefficient display, not purported re-trained trajectories.

Balanced objective, following `coati/_core/Epoch.py`:

`L = (1-c) A_X + c A_Y + lambda_X S_X + lambda_Y S_Y`

`A_m = lambda_E E_m + lambda_D D_m`

`E_X = 1/2 E[integral ||dx/dt||^2 dt]`, `E_Y = 1/2 E[integral ||dy/dt||^2 dt]`.

These energies are expectations over initial primary samples, averaged in implementation. `S_m` denotes the sum of per-observed-time Sinkhorn discrepancies. `D_m` denotes the implementation's manifold-density penalty. Terms that are disabled by configuration are omitted. Time discretization and sampling remain in the engine, not in this drawing script.

The static paper overview remains available as a separate PDF. No algorithm files are modified by this animation.
