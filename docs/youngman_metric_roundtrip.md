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
