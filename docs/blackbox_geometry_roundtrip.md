# Black-box geometry round trip

The black-box demo enforces the reconstruction boundary that the earlier
metric demo only approximated:

```text
expanded source geometry
  -> YoungMan crossings
  -> FIFO surface spline publication
  -> source access ends
  -> curvature-adaptive conforming triangulation of the spline callable
  -> mesh-backed cotangent/DEC geometry
  -> discrete Laplace--Beltrami comparison against source reference
```

The source is allowed to answer YoungMan queries. YoungMan itself carries the
expanded five-dimensional values across the solver-export boundary alongside
each crossing control. Spline publication consumes only that exported batch;
a regression test disables the source before fitting. The adaptive
triangulator receives that callable and a finite-difference Jacobian; it has no
source transform, implicit field, YoungMan topology, or source triangles.

This demo uses a single `(u,v)` graph chart. Multi-sheet surfaces will require
multiple charts or an atlas and are intentionally rejected by this design
rather than being silently projected onto one sheet.

## Stage certificates

The report separates:

- YoungMan edge-interpolation error;
- spline position and induced `2 x 2` metric error;
- triangulator chord and tangent tolerances;
- continuous spline-versus-source surface Laplacian error;
- mesh-versus-continuous-spline discretization error; and
- final mesh-versus-source Laplacian error.

Boundary, degenerate, singular, and nonmanifold vertices are flagged and
excluded from headline interior RMS values. RMS quantities are weighted by
lumped surface area rather than mesh vertex density. Triangle CSV rows retain
the fields needed for separate OpenGL or headless error views.

The mesh-backed stage is not a second Laplace implementation. It reuses the
same cotangent topology builder as the established mesh/DEC operator, then
executes its numeric geometry and reductions through `AbstractTensor`.
Only integer connectivity is realized on the host; geometry, weights, mass,
flux, masks, and Laplace values remain on the selected backend and on the
gradient tape.

Run and render the final error:

```powershell
python -m src.common.tensors.youngman.blackbox_roundtrip_demo `
    --render-image blackbox_roundtrip.png
```

Select `--error-field spline`, `triangulation`, `metric`, or `laplace` to
inspect a particular transition. The final mesh uses all five embedding
channels for metric and cotangent calculations; only its first three channels
are sent to the ordinary renderer.

`--error-field geometry` removes the diverging error palette and presents the
reconstructed mesh with Pluck's ordinary lit material. `--manifold` selects
`ripple`, `banana`, `saddle`, or `twisted_ribbon`. These presets change the
five-dimensional values exported through YoungMan and reconstructed by the
spline; they are not post-solve renderer deformations.

An uncertified run exits with an error by default. `--allow-unconverged`
exists for deliberate failure diagnostics, while `--max-rounds` and
`--max-triangles` make those stress cases reproducible.

The default `--target-epsilon` is `1e-6`. It is specifically the maximum
sampled positional deviation between the spline callable and its
piecewise-affine mesh, in embedding-coordinate units. The adaptive
triangulator increases parallel refinement waves until that epsilon and the
independent tangent tolerance are met. `epsilon_ratio` reports measured
maximum divided by target; success requires a ratio no greater than one.
Resource caps remain explicit, and exhausting one reports nonconvergence
instead of weakening epsilon.

This epsilon does not rename spline-versus-source or Laplace error as mesh
error. Those distinct quantities remain in the report and may be much larger;
improving them requires increasing YoungMan sampling or changing the spline
and discrete operator, respectively.

## Time-varying profiling

`--time-value` phase-shifts the source geometry while preserving a periodic
round trip (`t=0` and `t=1` agree). `--animation output.gif` repeats the
complete YoungMan → spline → triangulation → reference → mesh-Laplacian solve
for every frame. It does not deform or recolor a cached solution.

Each run records wall-clock time for YoungMan extraction, FIFO fitting,
adaptive triangulation, continuous reference evaluation, mesh
transform/Laplacian construction, and error reporting. Pluck's ordinary mesh
renderer displays current, arithmetic-mean, and 95th-percentile timings beside
the surface. All runs, including the first warm-up, enter the rolling
statistics. Image rendering and GIF encoding are deliberately outside the
solver profile and CSV so display cost cannot be mistaken for numerical cost.

`--live` opens Pluck's ordinary OpenGL viewer. A worker repeatedly performs
the complete solve while the GL thread continuously renders the newest
completed mesh. Publication is latest-result-wins: an unpublished stale frame
may be dropped, but a partial solve is never displayed. The HUD reports
whether each published mesh met its certificates.

The live HUD also profiles the video path: CPU mesh preparation/upload, draw
submission, HUD update, buffer swap, full frame wall time, solve-to-display
latency, rendered-frame count, and total session wall time. It does not call
`glFinish`, so draw submission is correctly labelled as CPU time rather than
pretending to measure completed GPU work.

## Appropriate neural participation

The safest useful neural component is a proposal model, not a replacement for
the measured operator. Candidate inputs include the current parameter samples,
metric tags, prior refinement history, and `t`; candidate outputs include
initial spline coefficients, likely failing triangles, or an anisotropic edge
priority. The deterministic spline, triangulator, topology checks, and
Laplace comparison then certify or reject that proposal.

This creates a fair experiment with three separately profiled paths:

1. an unassisted baseline;
2. neural proposal generation plus deterministic correction; and
3. deterministic certification shared by both.

Report proposal time, correction time, total time, rejection rate, and final
error together. Never omit rejected proposals or move network warm-up outside
one path only. A particularly natural first target is refinement prediction:
it can reduce expensive black-box probes while leaving mesh values and all
acceptance decisions under the existing geometric certificate.

The first geometry-linked learner is now active by default. It builds the
network from `abstract_nn` modules, captures its forward path once as the
shared `FusedProgram`, and replays that same program for training and
inference on the selected backend. It trains on a cross-manifold corpus and
reports held-out validation, its exported autograd schedule, and independent
pilot/guided mesh certification. `--no-train` provides the unassisted timing
baseline; `--training-epochs` controls the assisted run. A learned mesh is
deployed only when the model passes held-out validation, the guided
triangulation converges, and a fresh source-relative Laplace certificate
improves. See `docs/abstract_geometry_training.md` for the numeric backend
boundaries, objective separation, and selected autograd contract.
