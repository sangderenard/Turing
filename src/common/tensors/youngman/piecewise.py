"""Streaming piecewise simplex splines with expanded-dimensional metrics.

The core map is generic ``d -> m``.  The principal integration is ``3 -> m``:
the first three output channels are ordinary spatial geometry while every
output channel contributes to the induced ``3 x 3`` metric tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from queue import Empty, Queue
from threading import Lock
from typing import Dict, Mapping, Optional

import numpy as np


def simplex_multi_indices(dimension: int, degree: int) -> np.ndarray:
    """Return all ``dimension + 1`` barycentric exponents summing to degree."""
    if dimension < 1 or degree < 1:
        raise ValueError("dimension and degree must be positive")

    rows: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            visit(prefix + (value,), remaining - value, slots - 1)

    visit((), degree, dimension + 1)
    return np.asarray(rows, dtype=np.int64)


def _multinomial_coefficients(indices: np.ndarray, degree: int) -> np.ndarray:
    coefficients = np.full(len(indices), factorial(degree), dtype=np.float64)
    for column in range(indices.shape[1]):
        coefficients /= np.asarray(
            [factorial(int(value)) for value in indices[:, column]],
            dtype=np.float64,
        )
    return coefficients


def _monomial(
    barycentric: np.ndarray, exponents: np.ndarray
) -> np.ndarray:
    values = np.ones(len(barycentric), dtype=np.float64)
    for axis, exponent in enumerate(exponents):
        if exponent:
            values *= barycentric[:, axis] ** int(exponent)
    return values


@dataclass(frozen=True)
class SimplexBezierPatch:
    """One polynomial simplex patch mapping an intrinsic domain to an embedding."""

    patch_id: int
    domain_vertices: np.ndarray
    degree: int
    coefficients: np.ndarray
    multi_indices: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.domain_vertices, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        indices = np.asarray(self.multi_indices, dtype=np.int64)
        dimension = vertices.shape[1]
        expected_controls = len(simplex_multi_indices(dimension, self.degree))
        if vertices.shape != (dimension + 1, dimension):
            raise ValueError("domain_vertices must describe one full simplex")
        if indices.shape != (expected_controls, dimension + 1):
            raise ValueError("multi_indices do not match patch dimension/degree")
        if coefficients.ndim != 2 or len(coefficients) != expected_controls:
            raise ValueError("coefficient count does not match simplex basis")
        object.__setattr__(self, "domain_vertices", vertices)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "multi_indices", indices)

    @property
    def intrinsic_dimension(self) -> int:
        return int(self.domain_vertices.shape[1])

    @property
    def embedding_dimension(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def control_point_count(self) -> int:
        return int(len(self.coefficients))

    def _local_coordinates(
        self, parameters: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.ndim != 2 or parameters.shape[1] != self.intrinsic_dimension:
            raise ValueError(
                f"parameters must have shape (N, {self.intrinsic_dimension})"
            )
        edges = self.domain_vertices[1:] - self.domain_vertices[0]
        inverse_edges = np.linalg.inv(edges)
        local = (parameters - self.domain_vertices[0]) @ inverse_edges
        barycentric = np.concatenate(
            (1.0 - local.sum(axis=1, keepdims=True), local), axis=1
        )
        return local, barycentric, inverse_edges

    def contains(self, parameters: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
        _, barycentric, _ = self._local_coordinates(parameters)
        return np.all(barycentric >= -tolerance, axis=1)

    def _basis_bundle(
        self, parameters: np.ndarray, *, include_hessian: bool = False
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        _, barycentric, inverse_edges = self._local_coordinates(parameters)
        sample_count = len(barycentric)
        control_count = len(self.multi_indices)
        dimension = self.intrinsic_dimension
        multipliers = _multinomial_coefficients(
            self.multi_indices, self.degree
        )
        lambda_gradient = np.concatenate(
            (
                -np.ones((1, dimension), dtype=np.float64),
                np.eye(dimension, dtype=np.float64),
            ),
            axis=0,
        )
        basis = np.empty((sample_count, control_count), dtype=np.float64)
        derivative_local = np.zeros(
            (sample_count, control_count, dimension), dtype=np.float64
        )
        hessian_local = (
            np.zeros(
                (sample_count, control_count, dimension, dimension),
                dtype=np.float64,
            )
            if include_hessian else None
        )

        for control, alpha in enumerate(self.multi_indices):
            coefficient = multipliers[control]
            basis[:, control] = coefficient * _monomial(barycentric, alpha)
            for source_axis, exponent in enumerate(alpha):
                if exponent == 0:
                    continue
                reduced = alpha.copy()
                reduced[source_axis] -= 1
                term = coefficient * exponent * _monomial(barycentric, reduced)
                derivative_local[:, control, :] += (
                    term[:, None] * lambda_gradient[source_axis]
                )
                if hessian_local is not None:
                    for second_axis, second_exponent in enumerate(alpha):
                        remaining = second_exponent - (
                            1 if second_axis == source_axis else 0
                        )
                        if remaining <= 0:
                            continue
                        reduced_twice = reduced.copy()
                        reduced_twice[second_axis] -= 1
                        second_term = (
                            coefficient
                            * exponent
                            * remaining
                            * _monomial(barycentric, reduced_twice)
                        )
                        hessian_local[:, control, :, :] += (
                            second_term[:, None, None]
                            * np.outer(
                                lambda_gradient[source_axis],
                                lambda_gradient[second_axis],
                            )
                        )

        derivative_global = np.einsum(
            "nck,ak->nca", derivative_local, inverse_edges
        )
        hessian_global = None
        if hessian_local is not None:
            hessian_global = np.einsum(
                "nckl,ak,bl->ncab",
                hessian_local,
                inverse_edges,
                inverse_edges,
            )
        return basis, derivative_global, hessian_global

    def evaluate(self, parameters: np.ndarray) -> np.ndarray:
        basis, _, _ = self._basis_bundle(parameters)
        return basis @ self.coefficients

    def jacobian(self, parameters: np.ndarray) -> np.ndarray:
        _, derivative, _ = self._basis_bundle(parameters)
        return np.einsum("ncd,cm->nmd", derivative, self.coefficients)

    def hessian(self, parameters: np.ndarray) -> np.ndarray:
        _, _, hessian = self._basis_bundle(parameters, include_hessian=True)
        assert hessian is not None
        return np.einsum("ncab,cm->nmab", hessian, self.coefficients)

    def spatial_position(
        self, parameters: np.ndarray, spatial_dimensions: int = 3
    ) -> np.ndarray:
        if self.embedding_dimension < spatial_dimensions:
            raise ValueError("embedding has fewer channels than spatial geometry")
        return self.evaluate(parameters)[..., :spatial_dimensions]

    def metric_tensor(
        self,
        parameters: np.ndarray,
        *,
        ambient_metric: Optional[np.ndarray] = None,
        spatial_only: bool = False,
        spatial_dimensions: int = 3,
    ) -> np.ndarray:
        jacobian = self.jacobian(parameters)
        if spatial_only:
            jacobian = jacobian[:, :spatial_dimensions, :]
        if ambient_metric is None:
            return np.einsum("nmi,nmj->nij", jacobian, jacobian)
        ambient = np.asarray(ambient_metric, dtype=np.float64)
        if ambient.ndim == 2:
            return np.einsum("nmi,mk,nkj->nij", jacobian, ambient, jacobian)
        if ambient.ndim == 3:
            return np.einsum("nmi,nmk,nkj->nij", jacobian, ambient, jacobian)
        raise ValueError("ambient_metric must be (m,m) or (N,m,m)")

    def collapsed_metric_components(
        self, parameters: np.ndarray, spatial_dimensions: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return full, spatial, and hidden-dimension metric contributions."""
        full = self.metric_tensor(parameters)
        spatial = self.metric_tensor(
            parameters,
            spatial_only=True,
            spatial_dimensions=spatial_dimensions,
        )
        return full, spatial, full - spatial


class SimplexBezierFactory:
    """Fit vector-valued simplex patches from values and optional Jacobians."""

    @staticmethod
    def fit(
        patch_id: int,
        domain_vertices: np.ndarray,
        parameters: np.ndarray,
        embedded_values: np.ndarray,
        *,
        degree: int = 3,
        jacobians: Optional[np.ndarray] = None,
        derivative_weight: float = 1.0,
        ridge: float = 1e-12,
    ) -> SimplexBezierPatch:
        vertices = np.asarray(domain_vertices, dtype=np.float64)
        parameters = np.asarray(parameters, dtype=np.float64)
        values = np.asarray(embedded_values, dtype=np.float64)
        dimension = vertices.shape[1]
        indices = simplex_multi_indices(dimension, degree)
        prototype = SimplexBezierPatch(
            patch_id=patch_id,
            domain_vertices=vertices,
            degree=degree,
            coefficients=np.zeros((len(indices), values.shape[1])),
            multi_indices=indices,
        )
        basis, derivative, _ = prototype._basis_bundle(parameters)
        design_parts = [basis]
        target_parts = [values]
        if jacobians is not None:
            jacobians = np.asarray(jacobians, dtype=np.float64)
            expected = (len(parameters), values.shape[1], dimension)
            if jacobians.shape != expected:
                raise ValueError(f"jacobians must have shape {expected}")
            for axis in range(dimension):
                design_parts.append(derivative[:, :, axis] * derivative_weight)
                target_parts.append(jacobians[:, :, axis] * derivative_weight)
        design = np.concatenate(design_parts, axis=0)
        targets = np.concatenate(target_parts, axis=0)
        if ridge > 0.0:
            design = np.concatenate(
                (design, np.sqrt(ridge) * np.eye(len(indices))), axis=0
            )
            targets = np.concatenate(
                (targets, np.zeros((len(indices), values.shape[1]))), axis=0
            )
        coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
        if rank < len(indices):
            raise ValueError(
                f"patch {patch_id} is underconstrained: rank {rank}, "
                f"need {len(indices)}"
            )
        return SimplexBezierPatch(
            patch_id=patch_id,
            domain_vertices=vertices,
            degree=degree,
            coefficients=coefficients,
            multi_indices=indices,
        )


@dataclass(frozen=True)
class PiecewiseSplineGeneration:
    """Immutable collection of fitted patches published as one generation."""

    generation: int
    patches: Mapping[int, SimplexBezierPatch]

    @property
    def control_point_count(self) -> int:
        return sum(patch.control_point_count for patch in self.patches.values())

    def locate(self, parameters: np.ndarray) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=np.float64)
        owners = np.full(len(parameters), -1, dtype=np.int64)
        for patch_id in sorted(self.patches):
            unclaimed = owners < 0
            if not np.any(unclaimed):
                break
            indices = np.flatnonzero(unclaimed)
            inside = self.patches[patch_id].contains(parameters[indices])
            owners[indices[inside]] = patch_id
        return owners

    def _dispatch(self, parameters: np.ndarray, method: str) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=np.float64)
        owners = self.locate(parameters)
        if np.any(owners < 0):
            raise ValueError("one or more parameters lie outside every patch")
        output = None
        for patch_id in np.unique(owners):
            mask = owners == patch_id
            values = getattr(self.patches[int(patch_id)], method)(parameters[mask])
            if output is None:
                output = np.empty((len(parameters), *values.shape[1:]), dtype=np.float64)
            output[mask] = values
        assert output is not None
        return output

    def evaluate(self, parameters: np.ndarray) -> np.ndarray:
        return self._dispatch(parameters, "evaluate")

    def jacobian(self, parameters: np.ndarray) -> np.ndarray:
        return self._dispatch(parameters, "jacobian")

    def metric_tensor(self, parameters: np.ndarray) -> np.ndarray:
        return self._dispatch(parameters, "metric_tensor")


@dataclass(frozen=True)
class PatchSampleBatch:
    patch_ids: np.ndarray
    parameters: np.ndarray
    embedded_values: np.ndarray
    jacobians: Optional[np.ndarray] = None


class StreamingPiecewiseSplineEngine:
    """FIFO-fed piecewise fitter with immutable, atomically published generations."""

    def __init__(
        self,
        domain_simplices: Mapping[int, np.ndarray],
        *,
        degree: int = 3,
        derivative_weight: float = 1.0,
        ridge: float = 1e-12,
        max_samples_per_patch: Optional[int] = None,
    ) -> None:
        self.domain_simplices = {
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in domain_simplices.items()
        }
        self.degree = degree
        self.derivative_weight = derivative_weight
        self.ridge = ridge
        self.max_samples_per_patch = max_samples_per_patch
        self.input_fifo: Queue[PatchSampleBatch] = Queue()
        self._samples: Dict[int, list[np.ndarray]] = {}
        self._values: Dict[int, list[np.ndarray]] = {}
        self._jacobians: Dict[int, list[np.ndarray]] = {}
        self._latest: Optional[PiecewiseSplineGeneration] = None
        self._generation_lock = Lock()

    def submit(
        self,
        patch_ids: np.ndarray,
        parameters: np.ndarray,
        embedded_values: np.ndarray,
        jacobians: Optional[np.ndarray] = None,
    ) -> None:
        patch_ids = np.array(patch_ids, dtype=np.int64, copy=True)
        parameters = np.array(parameters, dtype=np.float64, copy=True)
        values = np.array(embedded_values, dtype=np.float64, copy=True)
        derivatives = (
            None if jacobians is None
            else np.array(jacobians, dtype=np.float64, copy=True)
        )
        if patch_ids.ndim != 1 or parameters.ndim != 2 or values.ndim != 2:
            raise ValueError("patch_ids must be a vector and sample values matrices")
        if not (len(patch_ids) == len(parameters) == len(values)):
            raise ValueError("FIFO sample arrays need equal row counts")
        if derivatives is not None and len(derivatives) != len(parameters):
            raise ValueError("FIFO Jacobians need one row per sample")
        unknown = set(np.unique(patch_ids)) - set(self.domain_simplices)
        if unknown:
            raise ValueError(f"unknown patch IDs: {sorted(unknown)}")
        if len(parameters):
            self.input_fifo.put_nowait(
                PatchSampleBatch(patch_ids, parameters, values, derivatives)
            )

    @property
    def pending_batches(self) -> int:
        return self.input_fifo.qsize()

    @property
    def latest_generation(self) -> Optional[PiecewiseSplineGeneration]:
        with self._generation_lock:
            return self._latest

    def _retain_tail(self, values: np.ndarray) -> np.ndarray:
        if self.max_samples_per_patch is None:
            return values
        if self.max_samples_per_patch <= 0:
            raise ValueError("max_samples_per_patch must be positive")
        return values[-self.max_samples_per_patch :]

    def update(self) -> Optional[PiecewiseSplineGeneration]:
        drained: list[PatchSampleBatch] = []
        for _ in range(self.input_fifo.qsize()):
            try:
                drained.append(self.input_fifo.get_nowait())
            except Empty:
                break
        if not drained:
            return self.latest_generation

        touched: set[int] = set()
        for batch in drained:
            for patch_id in np.unique(batch.patch_ids):
                patch_id = int(patch_id)
                mask = batch.patch_ids == patch_id
                self._samples.setdefault(patch_id, []).append(batch.parameters[mask])
                self._values.setdefault(patch_id, []).append(batch.embedded_values[mask])
                if batch.jacobians is not None:
                    self._jacobians.setdefault(patch_id, []).append(
                        batch.jacobians[mask]
                    )
                touched.add(patch_id)

        previous = self.latest_generation
        patches = {} if previous is None else dict(previous.patches)
        for patch_id in sorted(touched):
            parameters = self._retain_tail(
                np.concatenate(self._samples[patch_id], axis=0)
            )
            values = self._retain_tail(
                np.concatenate(self._values[patch_id], axis=0)
            )
            jacobians = None
            if patch_id in self._jacobians:
                jacobians = self._retain_tail(
                    np.concatenate(self._jacobians[patch_id], axis=0)
                )
                if len(jacobians) != len(parameters):
                    jacobians = None
            self._samples[patch_id] = [parameters]
            self._values[patch_id] = [values]
            if jacobians is not None:
                self._jacobians[patch_id] = [jacobians]
            patches[patch_id] = SimplexBezierFactory.fit(
                patch_id,
                self.domain_simplices[patch_id],
                parameters,
                values,
                degree=self.degree,
                jacobians=jacobians,
                derivative_weight=self.derivative_weight,
                ridge=self.ridge,
            )

        generation_number = 1 if previous is None else previous.generation + 1
        generation = PiecewiseSplineGeneration(generation_number, patches)
        with self._generation_lock:
            self._latest = generation
        return generation


# ---------------------------------------------------------------------------
# Destreamed construction: a function -> an adaptively subdivided polyspline
# ---------------------------------------------------------------------------
# The streaming engine above fits patches to samples as they arrive.  The
# same patches can be built directly from a FUNCTION: each simplex samples
# the function at its own Bezier domain points (the principal lattice,
# unisolvent for the degree), fits exactly, and integrates exactly -- every
# degree-n Bernstein basis polynomial on a d-simplex T integrates to
# |T| / C(n + d, d).  Subdividing where a patch disagrees with its children
# gives an arbitrarily refined polyspline and a measured error, returned as
# one object: the generation (callable) plus its integral.


def simplex_volume(vertices: np.ndarray) -> float:
    """|T| of a d-simplex given its d+1 vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = vertices[1:] - vertices[0]
    return float(abs(np.linalg.det(edges)) / factorial(vertices.shape[1]))


def domain_points(vertices: np.ndarray, degree: int) -> np.ndarray:
    """The Bezier domain points sum_i (alpha_i / n) v_i for |alpha| = n."""
    vertices = np.asarray(vertices, dtype=np.float64)
    indices = simplex_multi_indices(vertices.shape[1], degree)
    return (indices / float(degree)) @ vertices


def patch_integral(patch: SimplexBezierPatch) -> np.ndarray:
    """Exact integral of a patch over its simplex, one value per channel."""
    d, n = patch.intrinsic_dimension, patch.degree
    binomial = factorial(n + d) / (factorial(n) * factorial(d))
    return simplex_volume(patch.domain_vertices) * patch.coefficients.sum(axis=0) / binomial


def fit_function_patch(function, vertices: np.ndarray, degree: int,
                       patch_id: int = 0) -> SimplexBezierPatch:
    """Interpolate ``function`` on one simplex at its own domain points."""
    points = domain_points(vertices, degree)
    values = np.asarray(function(points), dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return SimplexBezierFactory.fit(patch_id, vertices, points, values,
                                    degree=degree, ridge=0.0)


def bisect_simplex(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a simplex in two across the midpoint of its longest edge."""
    vertices = np.asarray(vertices, dtype=np.float64)
    count = len(vertices)
    best, pair = -1.0, (0, 1)
    for i in range(count):
        for j in range(i + 1, count):
            length = float(np.sum((vertices[i] - vertices[j]) ** 2))
            if length > best:
                best, pair = length, (i, j)
    i, j = pair
    middle = 0.5 * (vertices[i] + vertices[j])
    left, right = vertices.copy(), vertices.copy()
    left[j] = middle
    right[i] = middle
    return left, right


def kuhn_simplices(lower, upper) -> list[np.ndarray]:
    """The d! Kuhn simplices tiling the box [lower, upper]."""
    from itertools import permutations

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    d = len(lower)
    out = []
    for order in permutations(range(d)):
        corner = lower.copy()
        vertices = [corner.copy()]
        for axis in order:
            corner[axis] = upper[axis]
            vertices.append(corner.copy())
        out.append(np.asarray(vertices))
    return out


def box_cells(lower, upper, breakpoints=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a box at declared breakpoints (kinks, singular values) per axis."""
    from itertools import product

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    breakpoints = breakpoints or {}
    edges = []
    for axis in range(len(lower)):
        cuts = sorted(float(c) for c in breakpoints.get(axis, ())
                      if lower[axis] < float(c) < upper[axis])
        edges.append([lower[axis], *cuts, upper[axis]])
    cells = []
    for index in product(*(range(len(e) - 1) for e in edges)):
        lo = np.asarray([edges[a][k] for a, k in enumerate(index)])
        hi = np.asarray([edges[a][k + 1] for a, k in enumerate(index)])
        cells.append((lo, hi))
    return cells


@dataclass(frozen=True)
class AdaptivePolyspline:
    """An adaptively subdivided polyspline of a function, and its integral.

    ``generation`` is callable (evaluate / jacobian / metric_tensor) over the
    whole domain; ``integral`` is the exact integral of the polyspline;
    ``error_estimate`` is the sum over leaves of |parent - children|, the
    measured disagreement at the last refinement of each leaf.
    """

    generation: PiecewiseSplineGeneration
    integral: np.ndarray
    error_estimate: float
    degree: int

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        return self.generation.evaluate(parameters)


def adaptive_polyspline(function, lower, upper, *, degree: int = 4,
                        epsilon: float = 1e-12, relative: float = 1e-12,
                        breakpoints=None, max_patches: int = 20000) -> AdaptivePolyspline:
    """Fit and integrate ``function`` over the box [lower, upper].

    ``function`` takes points of shape (N, d) and returns (N,) or (N, m).
    Declared ``breakpoints`` ({axis: [values]}) split the box before any
    fitting, so a kink never sits inside a patch.  Refinement bisects the
    leaf whose own integral disagrees most with its two children's, until
    the total disagreement is under max(epsilon, relative * |integral|).
    """
    import heapq

    leaves = []                     # heap of (-error, id, vertices, patch, children)
    counter = [0]

    def evaluate_leaf(vertices):
        patch = fit_function_patch(function, vertices, degree, counter[0])
        counter[0] += 1
        left, right = bisect_simplex(vertices)
        children = (fit_function_patch(function, left, degree, counter[0]),
                    fit_function_patch(function, right, degree, counter[0] + 1))
        counter[0] += 2
        whole = patch_integral(patch)
        split = patch_integral(children[0]) + patch_integral(children[1])
        # Two independent indicators, and a leaf is only as settled as the
        # worse of them: h (the parent against its bisected children) and p
        # (degree n against degree n+1 on the same simplex, which samples
        # different points).  The h indicator alone is fooled when parent and
        # children agree by coincidence -- a separable periodic integrand on
        # symmetric Kuhn cells does exactly that, leaving a coarse leaf
        # "converged" with the total off by 1e-3 against an estimate of 1e-7.
        richer = patch_integral(fit_function_patch(function, vertices, degree + 1, counter[0]))
        counter[0] += 1
        error = float(max(np.max(np.abs(whole - split)), np.max(np.abs(richer - split))))
        return error, children, split

    for lo, hi in box_cells(lower, upper, breakpoints):
        for simplex in kuhn_simplices(lo, hi):
            error, children, split = evaluate_leaf(simplex)
            heapq.heappush(leaves, (-error, counter[0], children, split))

    def totals():
        integral = sum(item[3] for item in leaves)
        error = sum(-item[0] for item in leaves)
        return integral, error

    integral, error = totals()
    while error > max(epsilon, relative * float(np.max(np.abs(integral)))):
        if 2 * len(leaves) > max_patches:
            break
        _, _, children, _ = heapq.heappop(leaves)
        for child in children:
            child_error, grandchildren, split = evaluate_leaf(child.domain_vertices)
            heapq.heappush(leaves, (-child_error, counter[0], grandchildren, split))
        integral, error = totals()

    patches = {}
    for item in leaves:
        for child in item[2]:
            patches[len(patches)] = child
    generation = PiecewiseSplineGeneration(generation=0, patches=patches)
    return AdaptivePolyspline(generation=generation, integral=np.asarray(integral),
                              error_estimate=float(error), degree=int(degree))
