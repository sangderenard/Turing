# Adaptive surface triangulation

`riemann.AdaptiveSurfaceTriangulator` is a geometry primitive independent of
YoungMan, splines, rendering, and any particular tensor backend. It accepts a
vectorized parametric callable

```text
F: (N, 2) parameters -> (N, m) embedded values
```

and produces a conforming triangle mesh. The first three embedding channels
can be rendered, while every channel participates in its curvature
certificate.

Refinement proceeds in parallel waves. Every unique edge midpoint required by
the current generation is evaluated as a batch. All triangles exceeding the
position or optional tangent tolerance select refinement edges together.
Shared edges have one canonical identity, and a split is propagated to both
incident triangles; recursive local subdivision handles triangles receiving
multiple splits without T-junctions.

Position error is the distance between the true embedded edge midpoint and
the linear chord midpoint. When a vectorized Jacobian is supplied, tangent
error measures deviation of the midpoint Jacobian from endpoint
interpolation. The latter provides the first-derivative control needed before
using a mesh for metric or Laplace–Beltrami work.

The returned generation is immutable and includes parameter vertices,
expanded-dimensional vertices, topology, per-triangle errors, evaluation
count, convergence state, and the stopping reason. `batch_size` can bound
memory; leaving it unset exposes the maximum data-parallel batch for each
wave.

This is the foundation, not the final mesh optimizer. Planned layers include
anisotropic edge selection from the induced metric, mesh-quality relaxation,
topology constraints, singularity/excision policies, and direct
`AbstractTensor` execution. Those extensions should preserve the callable and
generation contracts rather than coupling the engine to one demo.

## Relationship to existing geometry and DEC

The module lives in Turing's existing `riemann` package beside
`ManifoldPackage` and `GeometryFactory`; it does not establish a second
geometry subsystem. A generation exposes canonical `dec_edges` and a
`dec_faces` mapping matching the topology inputs expected by the DEC scaffold
in `laplace_nd`.

The existing DEC code builds incidence and approximate Hodge operators from a
supplied topology. It does not perform adaptive triangulation, and its face
discovery and higher-simplex support remain provisional. The triangulator
therefore produces topology for DEC rather than duplicating a DEC operator.

Guardian Geometry remains the natural eventual native implementation: its
design already names parametric domains, granular domains, meshes, DEC,
stencils, and parallel execution. Those particular mesh/granular components
are presently scaffolds, so this tested Python implementation can serve as an
executable specification for a later vectorized C++ port.
