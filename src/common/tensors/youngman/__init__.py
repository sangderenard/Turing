"""Backend-agnostic YoungMan-style parametric cell extraction."""

from .algorithm import (
    DomainTetrahedra,
    ExtractionResult,
    MetricSampleTagBatch,
    SolverSampleBatch,
    compile_grid_domain,
    extract_isosurface,
    metric_sample_tags,
    tetrahedra_from_grid_domain,
)
from .spline import (
    ControlPointBatch,
    ParametricSpline,
    SplineFactory,
    StreamingSplineSolver,
)
from .piecewise import (
    PatchSampleBatch,
    PiecewiseSplineGeneration,
    SimplexBezierFactory,
    SimplexBezierPatch,
    StreamingPiecewiseSplineEngine,
    simplex_multi_indices,
)

__all__ = [
    "ControlPointBatch",
    "DomainTetrahedra",
    "ExtractionResult",
    "MetricSampleTagBatch",
    "ParametricSpline",
    "PatchSampleBatch",
    "PiecewiseSplineGeneration",
    "SimplexBezierFactory",
    "SimplexBezierPatch",
    "SolverSampleBatch",
    "SplineFactory",
    "StreamingSplineSolver",
    "StreamingPiecewiseSplineEngine",
    "compile_grid_domain",
    "extract_isosurface",
    "metric_sample_tags",
    "simplex_multi_indices",
    "tetrahedra_from_grid_domain",
]
