# YoungMan metric round trip

The metric round-trip demo makes a deliberately nontrivial geometry pass
through the full experimental path:

1. a rippled parametric map embeds a three-dimensional domain in five
   dimensions;
2. YoungMan extracts an implicit surface while retaining parametric triangle
   provenance and every active-edge solver movement;
3. each movement receives the complete induced `3 x 3` matrix
   `g = J.transpose() @ J`, its inverse, determinant, and tetrahedron identity;
4. YoungMan movements and interior support samples enter the piecewise spline
   engine through its FIFO;
5. the spline reconstructs all five embedding channels and derives its own
   metric matrix;
6. the same scalar probe is evaluated with the Laplace–Beltrami operator under
   the source and reconstructed metrics; and
7. Pluck's ordinary OpenGL renderer displays the reconstructed surface with
   the signed difference as a blue–neutral–red material heat map.

The matrix sidecar does not pretend that a metric is a reversible encoding of
the expanded embedding. The five coordinates remain present in the spline,
while the matrix is copied explicitly for geometry-aware routing, refinement,
and later neural-network work. This preserves both representations instead of
silently choosing one and losing information.

Run a headless report:

```powershell
python -m src.common.tensors.youngman.metric_roundtrip_demo
```

Open the Pluck view:

```powershell
python -m src.common.tensors.youngman.metric_roundtrip_demo --view
```

The heat map is a useful error instrument, not just presentation. Blue means
the reconstructed Laplacian is lower than the source value, red means it is
higher, and the window caption reports the symmetric clipping range. A finer
grid or more support samples should reduce error, while high-curvature regions
remain visible as candidates for adaptive subdivision.

## Experimental boundary handling

`--resolve-boundaries` enables a deliberately isolated boundary experiment:

```powershell
python -m src.common.tensors.youngman.metric_roundtrip_demo `
    --resolve-boundaries --boundary-condition dirichlet
```

It uses `laplace_nd`'s face ordering `(u-, u+, v-, v+, w-, w+)` and honors
`GridDomain.grid_boundaries`. Interior points retain centered differences.
Included Dirichlet faces use a second-order one-sided flux derivative.
Neumann currently means prescribed zero normal flux. The report and triangle
table identify affected vertices, so boundary treatment is never silently
mixed into the interior error certificate.

This adapter does not yet call `BuildLaplace3D.build_general_laplace`. That
builder's boundary/index machinery is only partly covered by tests, and the
demo operates at irregular YoungMan crossings rather than a complete regular
grid. Periodic resolution is intentionally not offered because it additionally
requires compatible geometry and cross-seam spline adjacency.

## Singularity lineage and present policy

The Kakarot implementation contained an early singularity branch. It detected
only exactly zero directional metric terms, then applied a global
Dirichlet/Neumann choice or interpreted a boolean tensor as the choice at each
singular grid point. The later AbstractTensor version retains the arguments and
a correctly shaped `singularity_mask`, but currently initializes that mask to
all false. Its callback behavior therefore has not yet survived as an active,
tested resolution path.

This demo improves **detection**, without claiming to have solved replacement:
it flags non-finite matrices, small determinants, small eigenvalues, and
excessive condition numbers for both source and spline metrics. Counts appear
in the summary and triangle-level flags appear in the exported table. A later
resolver can use those flags to select Dirichlet, Neumann, excision, coordinate
remapping, or local refinement; until that policy is explicit, singular values
must not be silently regularized into apparently trustworthy Laplacians.
