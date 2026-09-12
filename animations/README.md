# COATI geometric introduction

A silent Manim film: two initial Gaussian groups and five terminal groups in a 2D primary space are mapped to a 3D surface with Gaussian bumps. The primary space is the coordinate projection of that surface. The film finishes with illustrative failures without reference, manifold and mass constraints.

## Reproduce

```bash
python -m venv .venv-manim
source .venv-manim/bin/activate
pip install -r animations/requirements.txt
python animations/geometry.py
python animations/render.py
```

Use `--preview` for 720p; the default is 1080p. Native Cairo/Pango libraries are required. Formulas use Matplotlib mathtext, with no system LaTeX dependency.

## Geometry and optimization

`T(x1,x2) = (x1,x2,h(x1,x2))`, where `h` is a sum of three Gaussian bumps. Projection onto the first two coordinates recovers the primary space. These bumps encode **extra geometry**, not the density of the observed groups. The blue and orange endpoint contours represent multiple Gaussian populations; orange contours are their mapped images on the surface.

For each of five paths from distinct points within the two initial groups, `geometry.py` minimizes the discrete action `(1-Cy) A_X[x] + Cy A_Y[T(x)]`. Each action is one half the sum of squared consecutive displacements divided by the time step. This uses mapped positions; no secondary velocity model or velocity formula is shown.

The optimizer fixes progress along each start–end direction on a uniform grid of 65 points and optimizes 63 interior transverse offsets, with endpoints fixed. It tries three initial paths and keeps the lowest objective. This is a restricted numerical geometry illustration, **not a COATI Neural ODE training run or a claim of a global geodesic optimum**. At Cy=0, paths are straight. At Cy=1, the paths detour and have lower secondary action than the straight paths. Seventeen coefficient settings are optimized; intermediate animation frames interpolate them. The paths start at distinct nearby states, so the illustration does not split one identical Neural ODE initial state. Moving particles illustrate these paths with small offsets, not independently optimized particle trajectories.

The Neural ODE equation connects the illustration to COATI. In the film's geometric comparison the reference endpoints are fixed, and only the action tradeoff is varied. The end screen's full objective is schematic paper notation: reference, manifold and mass coefficients are absorbed into their loss symbols. Mass applies only to the growth/unbalanced extension. No claim is made that Cy disables independently weighted reference constraints in the implementation.

## Final ablation panels

- Without reference matching: miss observed distributions.
- Without manifold support: take a shortcut through low-density regions.
- Without mass matching: reach plausible positions with incorrect population mass; circle area denotes mass.

These are possible failure schematics, not measured ablation outcomes. No training code or original TraInf files are modified.
