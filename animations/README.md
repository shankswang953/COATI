# COATI geometric introduction

A silent Manim film: two initial Gaussian groups and five terminal groups in a 2D primary space are mapped to a 3D surface with Gaussian bumps. The primary space is the coordinate projection of that surface. The text and numerical labels use Arial; formulas use an Arial-based math configuration with symbol fallback. An elevated oblique camera makes the 3D detours visible without a wireframe. The film finishes with illustrative failures without reference, manifold and mass constraints.

## Reproduce

```bash
python -m venv .venv-manim
source .venv-manim/bin/activate
pip install -r animations/requirements.txt
python animations/geometry.py
python animations/render.py
```

Use `--preview` for 720p; the default is 1080p. Native Cairo/Pango libraries and the Arial font are required. Prose is laid out using the actual Arial font file and converted to one combined path per line, preserving kerning and spaces. Subtitles crossfade as complete lines. Formulas use Matplotlib mathtext, with no system LaTeX dependency.

## Geometry and optimization

`T(x1,x2) = (x1,x2,h(x1,x2))`, where `h` is a sum of three Gaussian bumps. Projection onto the first two coordinates recovers the primary space. These bumps encode **extra geometry**, not the density of the observed groups. Contours are computed from the sum of all Gaussian densities at each observed time, at three shared density levels. The lowest level forms a single enclosing loop, while higher levels resolve the component peaks; they are not superimposed component ellipses. Secondary curves are the mapped primary level sets, not an assertion about equal density relative to 3D surface area. The blue and orange endpoint contours represent multiple Gaussian populations, with staggered terminal centers; orange contours are their mapped images on the surface.

For each of five paths from distinct points within the two initial groups, `geometry.py` minimizes the discrete action `(1-Cy) A_X[x] + Cy A_Y[T(x)]`. Each action is one half the sum of squared consecutive displacements divided by the time step. This uses mapped positions; no secondary velocity model or velocity formula is shown.

The optimizer fixes progress along each start–end direction on a uniform grid of 65 points and optimizes 63 interior transverse offsets, with endpoints fixed. It tries three initial paths and keeps the lowest objective. This is a restricted numerical geometry illustration, **not a COATI Neural ODE training run or a claim of a global geodesic optimum**. At Cy=0, paths are straight. At Cy=1, the paths detour and have lower secondary action than the straight paths. Seventeen coefficient settings are optimized; intermediate animation frames interpolate them. The paths start at distinct nearby states, so the illustration does not split one identical Neural ODE initial state. Moving particles illustrate these paths with small offsets, not independently optimized particle trajectories.

## Fitted neural field and slow integration

A separate illustrative MLP is fitted to the optimized toy paths (`train_field.py`), then refined against RK4 rollouts (`refine_field.py`). This is supervised fitting of a geometric illustration, **not COATI's joint distribution-training procedure**. The tiny fitted weights are included in `field_weights.npz`; rendering uses only NumPy inference (`neural_field.py`). Run `python animations/neural_field.py` to check arrival at the five target centers. Regeneration of weights requires PyTorch; it is not required to render the included weights.

The network diagram illustrates parameter fitting; its pulses are not a recorded optimization history. The following arrows are actual evaluations of the fitted time-dependent field near its sampled trajectories. The final five moving cells and growing traces come from RK4 integration of that field, played over 14 seconds. The time label is simulation time, not wall-clock time. This segment has no added particle offsets. The Gaussian groups have distinct anisotropic covariance shapes and angles; the same shapes are mapped into 3D. The first geometric comparison remains a separate coefficient demonstration.

The Neural ODE equation connects the illustration to COATI. In the film's geometric comparison the reference endpoints are fixed, and only the action tradeoff is varied. The end screen's full objective is schematic paper notation: reference, manifold and mass coefficients are absorbed into their loss symbols. Mass applies only to the growth/unbalanced extension. No claim is made that Cy disables independently weighted reference constraints in the implementation.

## Final ablation panels

- Without reference matching: miss observed distributions.
- Without manifold support: take a shortcut through low-density regions.
- Without mass matching: reach plausible positions with incorrect population mass; cell abundance indicates mass.

These are possible failure schematics, not measured ablation outcomes. No training code or original TraInf files are modified.

The film keeps explanations in the scene subtitles and removes persistent footnotes and group-count labels. The map is shown abstractly as `T: X → Y`; its concrete Gaussian height function is documented here and in `geometry.py`, not overlaid on the video.

The three final ablations run sequentially. Paths are drawn while cells move along them; the mass example grows a local group by visible divisions before only a few cells travel onward. No bottom captions or persistent footnotes are shown.

## Logo coda

A separate symbolic ending grows blue and orange branches from one cell, curls them into the C motif, and introduces paired alignment strands. The original cell boundary becomes O; the aligned strands and dashed links become the two-color A; T and I complete the wordmark. This is a brand metaphor, not a branching solution of the deterministic Neural ODE. No explanatory footers are added.
