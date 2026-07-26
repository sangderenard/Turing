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

The source is allowed to answer YoungMan queries and provide expanded
five-dimensional values for the exported crossing controls. The published
surface spline contains only its fitted interpolator. The adaptive
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

Boundary and degenerate vertices are flagged and excluded from headline
interior RMS values. Triangle CSV rows retain the fields needed for separate
OpenGL or headless error views.

Run and render the final error:

```powershell
python -m src.common.tensors.youngman.blackbox_roundtrip_demo `
    --render-image blackbox_roundtrip.png
```

Select `--error-field spline`, `triangulation`, `metric`, or `laplace` to
inspect a particular transition. The final mesh uses all five embedding
channels for metric and cotangent calculations; only its first three channels
are sent to the ordinary renderer.
