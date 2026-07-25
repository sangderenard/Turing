# YoungMan AbstractTensor demo

This demo reconstructs a compact numeric slice of the Kakarot/SpeakToMe
`YoungManAlgorithm` inside Turing.

It deliberately separates three responsibilities:

- `GridDomain` retains continuous parametric coordinates and defers their
  spatial resolution.
- YoungMan compiles the resolved structured domain into tetrahedral cells,
  evaluates an implicit signed difference through `AbstractTensor`, and
  interpolates all active edges in batches.
- Pluck's ordinary OpenGL `BaseGLRenderer` optionally presents the resulting
  triangle soup. The numerical core has no renderer dependency.

Run the analytical, topology-state, and performance tables:

```powershell
python -m src.common.tensors.youngman.demo --resolution 10 --repeats 3
```

The run also bulk-exports every active-edge interpolation made during the solve
as paired parametric and embedded control points. An unbounded FIFO accepts
those points without making the YoungMan producer wait for spline fitting. The
spline consumer drains a finite FIFO prefix, retains accumulated control points,
and atomically publishes the newly fitted model; arrivals during a fit remain
queued for the next update.

The spline experiment uses a two-dimensional surface chart in a
three-dimensional parameter domain and reconstructs it in a three-dimensional
warped embedding. Held-out control points are resolved through the original
domain transform to measure mean, maximum, and total squared reconstruction
error. It separately reports the linearization error introduced by YoungMan's
edge interpolation.

Write all pandas tables as CSV:

```powershell
python -m src.common.tensors.youngman.demo `
  --resolution 10 `
  --output-dir youngman_demo_output
```

Open the mesh using the sibling `spectral-analyzer` repository's ordinary
OpenGL renderer:

```powershell
python -m src.common.tensors.youngman.demo --resolution 10 --view
```

Set `PLUCK_ROOT` if `spectral-analyzer` is not a sibling of `turing`.
`--view-frames N` closes the window automatically after `N` frames and is
useful for smoke tests.

The mesh portion uses an analytical sphere signed difference. The spline
portion uses a smooth height field inside a warped parametric domain so that
the intrinsic, parameter, and embedding dimensions are explicit and the
original transform supplies an independent reconstruction target. Fields are
queried only after final cell positions are known.

The implementation also closes two `AbstractTensor` gaps exposed by exercising
the recovered Laplace geometry:

- Python scalar indices now work with `AbstractTensor.unravel_index`.
- zero-dimensional reduction results remain usable in arithmetic, including
  the domain extent calculation `U.max() - U.min()`.

The existing `VolumeMapGenerator` remains incomplete for arbitrary edge/face
graphs. This demo supplies a structured-grid tetrahedral decomposition rather
than claiming to solve generic volume detection.
