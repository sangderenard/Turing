"""Dimension-aware spline reconstruction from bulk YoungMan solver samples."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock
from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import RBFInterpolator

from ..abstraction import AbstractTensor


def validate_single_valued_chart(
    parameters: np.ndarray,
    *,
    intrinsic_axes: Sequence[int],
    tolerance: float = 1e-9,
) -> None:
    """Reject samples that project one chart location to conflicting parameters."""
    parameters = np.asarray(parameters, dtype=np.float64)
    axes = np.asarray(tuple(intrinsic_axes), dtype=np.int64)
    if parameters.ndim != 2 or len(axes) == 0:
        raise ValueError("parameters must be a matrix and axes non-empty")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    dependent = np.asarray(
        [axis for axis in range(parameters.shape[1]) if axis not in set(axes)]
    )
    if not len(dependent):
        return
    chart = parameters[:, axes]
    quantized = np.rint(chart / tolerance).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    for group in range(int(inverse.max()) + 1 if len(inverse) else 0):
        rows = parameters[inverse == group][:, dependent]
        if len(rows) > 1 and np.max(np.ptp(rows, axis=0)) > tolerance:
            raise ValueError(
                "surface samples are not single-valued in the selected chart"
            )


@dataclass(frozen=True)
class ParametricSpline:
    """A spline from an intrinsic parameter chart into an embedding space."""

    interpolator: RBFInterpolator
    parameter_dimension: int
    intrinsic_dimension: int
    embedding_dimension: int
    parameter_center: np.ndarray
    chart_basis: np.ndarray

    def chart(self, parameters: np.ndarray) -> np.ndarray:
        points = np.asarray(parameters, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.parameter_dimension:
            raise ValueError(
                f"parameters must have shape (N, {self.parameter_dimension})"
            )
        return (points - self.parameter_center) @ self.chart_basis

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        """Evaluate embedded positions for a bulk parameter array."""
        return np.asarray(self.interpolator(self.chart(parameters)), dtype=np.float64)


class SplineFactory:
    """Construct splines with independently declared chart and embedding sizes."""

    @staticmethod
    def fit(
        parameters: np.ndarray,
        embedded_values: np.ndarray,
        *,
        intrinsic_dimension: Optional[int] = None,
        intrinsic_axes: Optional[Sequence[int]] = None,
        smoothing: float = 1e-10,
        kernel: str = "thin_plate_spline",
        neighbors: Optional[int] = 64,
    ) -> ParametricSpline:
        parameters = np.asarray(parameters, dtype=np.float64)
        embedded_values = np.asarray(embedded_values, dtype=np.float64)
        if parameters.ndim != 2 or embedded_values.ndim != 2:
            raise ValueError("parameters and embedded_values must both be matrices")
        if len(parameters) != len(embedded_values) or len(parameters) == 0:
            raise ValueError("sample matrices must have the same non-zero row count")

        parameter_dimension = parameters.shape[1]
        embedding_dimension = embedded_values.shape[1]
        center = parameters.mean(axis=0)

        if intrinsic_axes is not None:
            axes = np.asarray(tuple(intrinsic_axes), dtype=np.int64)
            if axes.ndim != 1 or len(axes) == 0:
                raise ValueError("intrinsic_axes must select at least one axis")
            if np.any(axes < 0) or np.any(axes >= parameter_dimension):
                raise ValueError("intrinsic_axes contains an invalid parameter axis")
            basis = np.eye(parameter_dimension, dtype=np.float64)[:, axes]
            intrinsic_dimension = len(axes)
        else:
            if intrinsic_dimension is None:
                intrinsic_dimension = parameter_dimension
            if not 1 <= intrinsic_dimension <= parameter_dimension:
                raise ValueError("intrinsic_dimension is outside the parameter space")
            _, _, right = np.linalg.svd(parameters - center, full_matrices=False)
            basis = right[:intrinsic_dimension].T

        chart_points = (parameters - center) @ basis
        # Shared tetrahedral edges produce duplicate solves. Average their
        # targets so the spline receives one stable value per chart location.
        unique_chart, inverse = np.unique(chart_points, axis=0, return_inverse=True)
        unique_values = np.zeros(
            (len(unique_chart), embedding_dimension), dtype=np.float64
        )
        counts = np.bincount(inverse)
        np.add.at(unique_values, inverse, embedded_values)
        unique_values /= counts[:, None]
        minimum = intrinsic_dimension + 1
        if len(unique_chart) < minimum:
            raise ValueError(
                f"need at least {minimum} distinct samples for a "
                f"{intrinsic_dimension}D spline"
            )

        interpolator = RBFInterpolator(
            unique_chart,
            unique_values,
            smoothing=smoothing,
            kernel=kernel,
            neighbors=(
                None if neighbors is None else min(neighbors, len(unique_chart))
            ),
        )
        return ParametricSpline(
            interpolator=interpolator,
            parameter_dimension=parameter_dimension,
            intrinsic_dimension=intrinsic_dimension,
            embedding_dimension=embedding_dimension,
            parameter_center=center,
            chart_basis=basis,
        )


@dataclass(frozen=True)
class ControlPointBatch:
    """One FIFO message containing paired chart and embedding values."""

    parameters: np.ndarray
    embedded_values: np.ndarray


class StreamingSplineSolver:
    """FIFO-fed spline solver whose producer never waits for a fit.

    ``submit`` only validates and enqueues an immutable copy. ``update`` drains
    the batches that were present at its start and fits that stable snapshot.
    Producers may continue submitting while the fit runs; those later batches
    remain queued for the next update. The completed model is published with
    one reference swap.
    """

    def __init__(
        self,
        *,
        intrinsic_dimension: Optional[int] = None,
        intrinsic_axes: Optional[Sequence[int]] = None,
        smoothing: float = 1e-10,
        kernel: str = "thin_plate_spline",
        neighbors: Optional[int] = 64,
        max_control_points: Optional[int] = None,
    ) -> None:
        self.input_fifo: Queue[ControlPointBatch] = Queue()
        self.intrinsic_dimension = intrinsic_dimension
        self.intrinsic_axes = (
            None if intrinsic_axes is None else tuple(intrinsic_axes)
        )
        self.smoothing = smoothing
        self.kernel = kernel
        self.neighbors = neighbors
        self.max_control_points = max_control_points
        self._parameters: Optional[np.ndarray] = None
        self._embedded_values: Optional[np.ndarray] = None
        self._model: Optional[ParametricSpline] = None
        self._model_lock = Lock()

    def submit(
        self, parameters: np.ndarray, embedded_values: np.ndarray
    ) -> None:
        """Append one control-point batch without waiting for spline fitting."""
        parameters = np.array(parameters, dtype=np.float64, copy=True)
        embedded_values = np.array(embedded_values, dtype=np.float64, copy=True)
        if parameters.ndim != 2 or embedded_values.ndim != 2:
            raise ValueError("FIFO control-point values must be matrices")
        if len(parameters) != len(embedded_values):
            raise ValueError("FIFO control-point matrices need equal row counts")
        if len(parameters):
            self.input_fifo.put_nowait(
                ControlPointBatch(parameters, embedded_values)
            )

    def submit_solver_samples(self, samples) -> None:
        """Enqueue a YoungMan ``SolverSampleBatch`` without importing it here."""
        if samples.parametric_points is None:
            raise ValueError("solver samples do not contain parametric points")
        self.submit(samples.parametric_points, samples.embedded_points)

    @property
    def pending_batches(self) -> int:
        return self.input_fifo.qsize()

    @property
    def control_point_count(self) -> int:
        return 0 if self._parameters is None else len(self._parameters)

    @property
    def latest_model(self) -> Optional[ParametricSpline]:
        with self._model_lock:
            return self._model

    def update(self) -> Optional[ParametricSpline]:
        """Consume the current FIFO prefix, refit, and publish a new model."""
        drained: list[ControlPointBatch] = []
        # qsize is advisory under concurrency, but capturing it gives this
        # update a finite prefix: arrivals during fitting belong to the next
        # update and cannot starve the current one.
        for _ in range(self.input_fifo.qsize()):
            try:
                drained.append(self.input_fifo.get_nowait())
            except Empty:
                break
        if not drained:
            return self.latest_model

        new_parameters = np.concatenate(
            [batch.parameters for batch in drained], axis=0
        )
        new_values = np.concatenate(
            [batch.embedded_values for batch in drained], axis=0
        )
        if self._parameters is not None:
            new_parameters = np.concatenate(
                (self._parameters, new_parameters), axis=0
            )
            new_values = np.concatenate(
                (self._embedded_values, new_values), axis=0
            )
        if self.max_control_points is not None:
            if self.max_control_points <= 0:
                raise ValueError("max_control_points must be positive")
            new_parameters = new_parameters[-self.max_control_points :]
            new_values = new_values[-self.max_control_points :]

        model = SplineFactory.fit(
            new_parameters,
            new_values,
            intrinsic_dimension=self.intrinsic_dimension,
            intrinsic_axes=self.intrinsic_axes,
            smoothing=self.smoothing,
            kernel=self.kernel,
            neighbors=self.neighbors,
        )
        self._parameters = new_parameters
        self._embedded_values = new_values
        with self._model_lock:
            self._model = model
        return model


@dataclass(frozen=True)
class AbstractKernelInterpolator:
    """Thin-plate vector spline expressed entirely through AbstractTensor ops."""

    chart_controls: AbstractTensor
    radial_coefficients: AbstractTensor
    polynomial_coefficients: AbstractTensor
    regularization: float = 1e-10

    @staticmethod
    def _kernel(left: AbstractTensor, right: AbstractTensor) -> AbstractTensor:
        differences = (
            left.reshape(left.shape[0], 1, left.shape[1])
            - right.reshape(1, right.shape[0], right.shape[1])
        )
        squared_distance = (differences * differences).sum(dim=2)
        return squared_distance * (squared_distance + 1e-30).log() * 0.5

    @classmethod
    def fit(
        cls,
        parameters: np.ndarray,
        embedded_values: np.ndarray,
        *,
        intrinsic_axes: Sequence[int] = (0, 1),
        bandwidth: float | None = None,
    ) -> "AbstractKernelInterpolator":
        parameters = np.asarray(parameters, dtype=np.float64)
        values = np.asarray(embedded_values, dtype=np.float64)
        chart = parameters[:, tuple(intrinsic_axes)]
        unique_chart, inverse = np.unique(chart, axis=0, return_inverse=True)
        unique_values = np.zeros((len(unique_chart), values.shape[1]))
        counts = np.bincount(inverse)
        np.add.at(unique_values, inverse, values)
        unique_values /= counts[:, None]
        chart = unique_chart
        values = unique_values
        chart_tensor = AbstractTensor.tensor(chart, dtype="float64")
        value_tensor = AbstractTensor.tensor(values, dtype="float64")
        dtype = chart_tensor.get_dtype()
        device = chart_tensor.get_device()
        backend = type(chart_tensor)
        radial = cls._kernel(chart_tensor, chart_tensor)
        polynomial = AbstractTensor.cat(
            (
                AbstractTensor.ones(
                    (len(chart), 1), dtype=dtype, device=device, cls=backend
                ),
                chart_tensor,
            ),
            dim=1,
        )
        zero = AbstractTensor.zeros(
            (polynomial.shape[1], polynomial.shape[1]),
            dtype=dtype,
            device=device,
            cls=backend,
        )
        system = AbstractTensor.cat(
            (
                AbstractTensor.cat(
                    (
                        radial + 1e-10 * AbstractTensor.eye(
                            len(chart), dtype=dtype, device=device
                        ),
                        polynomial,
                    ),
                    dim=1,
                ),
                AbstractTensor.cat((polynomial.swapaxes(0, 1), zero), dim=1),
            ),
            dim=0,
        )
        rhs = AbstractTensor.cat(
            (
                value_tensor,
                AbstractTensor.zeros(
                    (polynomial.shape[1], value_tensor.shape[1]),
                    dtype=dtype,
                    device=device,
                    cls=backend,
                ),
            ),
            dim=0,
        )
        coefficients = AbstractTensor.linalg.solve(system, rhs)
        return cls(
            chart_tensor,
            coefficients[: len(chart)],
            coefficients[len(chart) :],
        )

    @property
    def embedding_dimension(self) -> int:
        return int(self.radial_coefficients.shape[-1])

    def __call__(self, chart_parameters) -> AbstractTensor:
        query = (
            chart_parameters.to_backend(self.chart_controls).to_dtype(
                self.chart_controls.get_dtype()
            )
            if isinstance(chart_parameters, AbstractTensor)
            else AbstractTensor.tensor(
                chart_parameters,
                dtype=self.chart_controls.get_dtype(),
                device=self.chart_controls.get_device(),
            )
        )
        radial = self._kernel(query, self.chart_controls)
        polynomial = AbstractTensor.cat(
            (
                AbstractTensor.ones(
                    (query.shape[0], 1),
                    dtype=query.get_dtype(),
                    device=query.get_device(),
                    cls=type(query),
                ),
                query,
            ),
            dim=1,
        )
        return (
            radial @ self.radial_coefficients
            + polynomial @ self.polynomial_coefficients
        )
