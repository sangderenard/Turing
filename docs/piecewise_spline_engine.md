# Streaming piecewise simplex spline engine

## Contract

The engine represents a piecewise map

```text
F : R^d -> R^m
```

with one polynomial Bézier patch per domain simplex. The implementation is
generic in intrinsic dimension `d`, embedding dimension `m`, and polynomial
degree. Its first geometry integration uses `d = 3` and `m >= 3`.

The complete `m`-dimensional map is authoritative. “Collapsed into 3D” means
that the first three output channels are exposed as ordinary spatial geometry;
it does not mean that the remaining channels are discarded.

For patch Jacobian `J`, the induced metric is

```text
g = J^T J
```

or, with an ambient metric `G`,

```text
g = J^T G J
```

The engine can report the full metric, the metric induced by the visible three
channels, and their difference. The difference records the local metric
contribution of hidden embedding dimensions. It is not a lossless replacement
for those dimensions, so the full embedding coefficients remain available.

## Streaming

The producer submits immutable control-point batches to an unbounded FIFO. An
update drains a finite prefix, accumulates samples per patch, fits the touched
patches, and atomically publishes an immutable generation. Samples that arrive
during fitting remain queued for the next generation.

Samples are retained without a limit by default. A caller can request a bounded
per-patch history, but doing so explicitly accepts loss of older fitting data.

## Derivatives and geometry

Every patch exposes batched:

- embedded value;
- first derivative/Jacobian;
- second derivative/Hessian;
- visible three-dimensional position;
- induced metric tensor;
- spatial-only and hidden-dimension metric contributions.

Analytic derivatives are taken from the barycentric polynomial basis. A fit can
use value samples alone or value and Jacobian constraints together.

The current fitter uses least squares and optional ridge regularization. A
zero-ridge cubic fit exactly reproduces a sufficiently sampled cubic target up
to floating-point error. General source data remains an approximation and
requires independent error certification before machining or simulation use.

## Intended YoungMan metric channel

YoungMan currently searches and interpolates a three-dimensional tetrahedral
complex using ordinary Euclidean edge geometry. The intended integration is to
bring the spline/domain metric directly into that process without requiring
YoungMan to stop being a three-dimensional search algorithm.

Each cell, active-edge solve, crossing point, or emitted surface vertex should
be able to carry a metric sidecar:

```text
MetricSampleTag
    metric                  symmetric 3 x 3 local tensor
    inverse_metric          optional cached inverse
    determinant             optional volume-density term
    parameter_position      point at which the metric was evaluated
    patch_and_generation    spline/domain provenance
    validity_region         local patch or cell identity
    error_certificate       tolerance and source-oracle information
```

The tag is attached by stable sample/cell identity rather than being packed
into the visible three spatial coordinates. This lets a shuttle, a refinement
worker, a renderer, or a later differential operator recover the local geometry
that existed in the expanded embedding.

The integration should proceed in stages:

1. **Annotate:** YoungMan keeps its existing Euclidean topology and case table,
   while exported solver samples receive local metric tags.
2. **Measure:** edge lengths, triangle quality, curvature tolerances, and shuttle
   travel cost can optionally be evaluated with the tagged metric.
3. **Refine:** metric length and anisotropy can influence which tetrahedra or
   surface edges split and in which direction.
4. **Solve:** after validation, metric-aware interpolation and traversal can
   participate directly in YoungMan's search rather than only advising later
   work.

Keeping annotation as the first stage is important. It preserves a directly
comparable Euclidean baseline and allows metric data to be recorded now even
before every consumer understands it.

For an edge displacement `dx`, the local squared length becomes

```text
length_squared = dx^T g dx
```

This gives the shuttle or algorithm a compact local instruction about how the
apparently three-dimensional cell is stretched, compressed, or sheared by the
full expanded-dimensional map. The metric does not reconstruct the hidden
coordinates; the full spline remains available whenever their actual values or
orientation are required.

## Demonstration

Run:

```powershell
python -m src.common.tensors.youngman.piecewise_demo
```

The demo divides a three-dimensional cube into six tetrahedral patches and
streams samples from a known cubic `3 -> 5` map. The first three channels form
visible geometry; two additional channels alter the full induced metric.

The report separately measures:

- expanded-space value error;
- visible three-dimensional error;
- Jacobian error;
- full metric error;
- the magnitude of the hidden-dimension metric contribution;
- FIFO and immutable-generation state;
- control-point counts per patch.

## Faithfulness

No fitted spline can recover source behavior that was never sampled or
constrained. A production tolerance system should therefore retain:

- source samples or an authoritative oracle identity;
- patch generation and parentage;
- value and Jacobian error certificates;
- continuity measurements at patch boundaries;
- fitting policy and regularization;
- an oracle fallback outside certified regions.

The piecewise spline is a queryable surrogate and acceleration structure, not a
destructive replacement for its source.
