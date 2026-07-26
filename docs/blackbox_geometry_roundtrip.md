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

Run and render the final error:

```powershell
python -m src.common.tensors.youngman.blackbox_roundtrip_demo `
    --render-image blackbox_roundtrip.png
```

Select `--error-field spline`, `triangulation`, `metric`, or `laplace` to
inspect a particular transition. The final mesh uses all five embedding
channels for metric and cotangent calculations; only its first three channels
are sent to the ordinary renderer.

An uncertified run exits with an error by default. `--allow-unconverged`
exists for deliberate failure diagnostics, while `--max-rounds` and
`--max-triangles` make those stress cases reproducible.

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
