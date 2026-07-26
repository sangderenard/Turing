"""Truthful source -> YoungMan -> spline -> mesh -> Laplace round trip."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from ..abstract_convolution.laplace_nd import (
    GridDomain,
    continuous_laplace_beltrami,
)
from ..abstraction import AbstractTensor
from ..riemann import (
    AdaptiveSurfaceTriangulator,
    TriangulatedSurfaceTransform,
    TriangulationTolerance,
    abstract_mesh_laplace,
)
from .algorithm import DomainTetrahedra, compile_grid_domain, extract_isosurface
from .metric_roundtrip_demo import detailed_embedding, detailed_jacobian
from .refinement_network import (
    train_refinement_predictor,
    triangle_refinement_features,
    triangle_spring_edges,
)
from .spline import AbstractKernelInterpolator, validate_single_valued_chart


@dataclass(frozen=True)
class BlackBoxRoundTrip:
    summary: pd.DataFrame
    triangles: pd.DataFrame
    mesh: object
    geometry_transform: TriangulatedSurfaceTransform
    profile: pd.DataFrame
    training: pd.DataFrame


def deployment_decision(
    *, model_accepted: bool, guided_converged: bool, objective_improved: bool
) -> tuple[bool, str]:
    """Apply all independent gates required to promote a learned mesh."""
    if not model_accepted:
        return False, "model_not_accepted"
    if not guided_converged:
        return False, "guided_mesh_unconverged"
    if not objective_improved:
        return False, "laplace_not_improved"
    return True, "accepted"


@dataclass(frozen=True)
class PublishedSurfaceSpline:
    """The source-free callable handed across the triangulation boundary."""

    model: object

    def __call__(self, uv):
        if isinstance(uv, AbstractTensor):
            return self.model(uv)
        return self.model(np.asarray(uv, dtype=np.float64))

    def jacobian(self, uv):
        tensor_input = isinstance(uv, AbstractTensor)
        uv = uv if tensor_input else AbstractTensor.tensor(uv, dtype="float64")
        columns = []
        for axis in range(uv.get_shape()[-1]):
            offset = AbstractTensor.zeros_like(uv)
            offset[:, axis] = 2e-5
            plus = self.model(uv + offset)
            minus = self.model(uv - offset)
            columns.append((plus - minus) / (4e-5))
        result = AbstractTensor.stack(columns, dim=2)
        return result if tensor_input else result


def source_surface_parameters(
    uv, time_value: float = 0.0
):
    """Exact source chart, used only by YoungMan/reference measurements."""
    tensor_input = isinstance(uv, AbstractTensor)
    uv = uv if tensor_input else AbstractTensor.tensor(uv, dtype="float64")
    u, v = uv[:, 0], uv[:, 1]
    tau = 2.0 * np.pi
    phase = tau * float(time_value)
    target_z = 0.5 + 0.12 * (tau * u + phase).sin() * (tau * v).cos()
    visible_warp = 0.075 * (tau * u + phase).sin() * (tau * v).sin()
    result = AbstractTensor.stack((u, v, target_z - visible_warp), dim=1)
    return result if tensor_input else _host(result)


def manifold_embedding(
    parameters,
    time_value: float = 0.0,
    manifold: str = "ripple",
) :
    """Embed the common parameter volume into selectable visible manifolds."""
    tensor_input = isinstance(parameters, AbstractTensor)
    parameters = (
        parameters
        if tensor_input
        else AbstractTensor.tensor(parameters, dtype="float64")
    )
    u, v, w = parameters[:, 0], parameters[:, 1], parameters[:, 2]
    tau = 2.0 * np.pi
    base = (
        u,
        v,
        w + 0.075 * (tau * u).sin() * (tau * v).sin(),
        0.11
        * (2.0 * tau * u).sin()
        * (1.5 * tau * v).cos()
        * (1.0 + 0.2 * w),
        0.09 * (1.5 * tau * u).cos() * (2.0 * tau * w).sin()
        + 0.035 * (2.5 * tau * v).sin(),
    )
    if manifold == "ripple":
        result = AbstractTensor.stack(base, dim=1)
        return result if tensor_input else _host(result)
    phase = 2.0 * np.pi * float(time_value)
    height = base[2] - 0.5
    if manifold == "banana":
        angle = 2.35 * (u - 0.5) + 0.2 * (2.0 * np.pi * v + phase).sin()
        radius = 1.25 + 0.42 * (v - 0.5)
        visible = (
            radius * angle.sin(),
            1.35 * (v - 0.5),
            radius * angle.cos() - 1.25 + 0.7 * height,
        )
    elif manifold == "saddle":
        x = 1.45 * (u - 0.5)
        y = 1.45 * (v - 0.5)
        visible = (x, y, 0.52 * (x * x - y * y) + height)
    elif manifold == "twisted_ribbon":
        x = 1.8 * (u - 0.5)
        across = 1.1 * (v - 0.5)
        angle = 2.0 * np.pi * (u + time_value)
        visible = (
            x,
            across * angle.cos() - height * angle.sin(),
            across * angle.sin() + height * angle.cos(),
        )
    else:
        raise ValueError(f"unknown manifold preset: {manifold}")
    result = AbstractTensor.stack((*visible, base[3], base[4]), dim=1)
    return result if tensor_input else _host(result)


def source_surface(
    uv,
    time_value: float = 0.0,
    manifold: str = "ripple",
):
    tensor_input = isinstance(uv, AbstractTensor)
    result = manifold_embedding(
        source_surface_parameters(uv, time_value), time_value, manifold
    )
    return result if tensor_input else _host(result)


def source_surface_jacobian(
    uv,
    time_value: float = 0.0,
    manifold: str = "ripple",
) :
    if isinstance(uv, AbstractTensor):
        columns = []
        for axis in range(uv.get_shape()[-1]):
            offset = AbstractTensor.zeros_like(uv)
            offset[:, axis] = 2e-5
            columns.append(
                (
                    source_surface(uv + offset, time_value, manifold)
                    - source_surface(uv - offset, time_value, manifold)
                )
                / 4e-5
            )
        return AbstractTensor.stack(columns, dim=2)
    if manifold != "ripple":
        return _finite_jacobian(
            lambda query: source_surface(query, time_value, manifold), uv
        )
    uv = np.asarray(uv, dtype=np.float64)
    u, v = uv.T
    tau = 2.0 * np.pi
    phase = tau * float(time_value)
    parameter_jacobian = np.zeros((len(uv), 3, 2), dtype=np.float64)
    parameter_jacobian[:, 0, 0] = 1.0
    parameter_jacobian[:, 1, 1] = 1.0
    parameter_jacobian[:, 2, 0] = (
        0.12 * tau * np.cos(tau * u + phase) * np.cos(tau * v)
        - 0.075 * tau * np.cos(tau * u + phase) * np.sin(tau * v)
    )
    parameter_jacobian[:, 2, 1] = (
        -0.12 * tau * np.sin(tau * u + phase) * np.sin(tau * v)
        - 0.075 * tau * np.sin(tau * u + phase) * np.cos(tau * v)
    )
    return np.einsum(
        "nmi,nij->nmj",
        detailed_jacobian(source_surface_parameters(uv, time_value)),
        parameter_jacobian,
    )


def _finite_jacobian(function, uv: np.ndarray, step: float = 2e-5) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    center = np.asarray(function(uv), dtype=np.float64)
    jacobian = np.empty((len(uv), center.shape[1], 2), dtype=np.float64)
    for axis in range(2):
        offset = np.zeros_like(uv)
        offset[:, axis] = step
        jacobian[:, :, axis] = (
            function(uv + offset) - function(uv - offset)
        ) / (2.0 * step)
    return jacobian


def _host(value) -> np.ndarray:
    if isinstance(value, AbstractTensor):
        value = value.tolist()
    return np.asarray(value, dtype=np.float64)


def _metric(jacobian):
    if isinstance(jacobian, AbstractTensor):
        return jacobian.swapaxes(1, 2) @ jacobian
    return np.einsum("nmi,nmj->nij", jacobian, jacobian)


def probe_values(uv: np.ndarray) -> np.ndarray:
    u, v = np.asarray(uv, dtype=np.float64).T
    tau = 2.0 * np.pi
    return np.sin(tau * u) * np.cos(tau * v) + 0.1 * u * v


def probe_gradient(uv):
    tensor_input = isinstance(uv, AbstractTensor)
    uv = uv if tensor_input else AbstractTensor.tensor(uv, dtype="float64")
    u, v = uv[:, 0], uv[:, 1]
    tau = 2.0 * np.pi
    result = AbstractTensor.stack(
        (
            tau * (tau * u).cos() * (tau * v).cos() + 0.1 * v,
            -tau * (tau * u).sin() * (tau * v).sin() + 0.1 * u,
        ),
        dim=1,
    )
    return result if tensor_input else _host(result)


def continuous_surface_laplace(uv, metric_function, step: float = 2e-5):
    """Compatibility name for the shared rank-N AbstractTensor operator."""
    tensor_input = isinstance(uv, AbstractTensor)
    coordinates = (
        uv if tensor_input else AbstractTensor.tensor(uv, dtype="float64")
    )
    result = continuous_laplace_beltrami(
        coordinates, metric_function, probe_gradient, step=step
    )
    return result if tensor_input else _host(result)


def _mesh_laplace_objective(
    mesh,
    time_value: float,
    manifold: str,
    *,
    spline_surface=None,
    component: str = "total",
):
    """Measure one independently defined Laplace-error component."""
    uv = mesh.parameters
    transformed = TriangulatedSurfaceTransform.from_mesh(
        uv, mesh.embedded, mesh.triangles
    ).laplace(probe_values(uv))
    reference = continuous_surface_laplace(
        uv,
        lambda query: _metric(
            source_surface_jacobian(query, time_value, manifold)
        ),
    )
    if component == "total":
        error = transformed.laplacian - reference
    else:
        if spline_surface is None:
            raise ValueError(f"{component} objective requires spline_surface")
        spline_reference = continuous_surface_laplace(
            uv,
            lambda query: _metric(spline_surface.jacobian(query)),
        )
        if component == "discretization":
            error = transformed.laplacian - spline_reference
        elif component == "reconstruction":
            error = spline_reference - reference
        else:
            raise ValueError(f"unknown Laplace component: {component}")
    valid = (
        ~transformed.geometry.boundary_vertex_mask
        & ~transformed.geometry.invalid_vertex_mask
        & np.isfinite(error)
    )
    weights = transformed.geometry.lumped_vertex_areas[valid]
    loss = (
        float(np.sqrt(np.sum(weights * error[valid] ** 2) / weights.sum()))
        if len(weights) and weights.sum() > 0.0
        else np.inf
    )
    finite = np.isfinite(error)[mesh.triangles]
    values = np.where(finite, np.abs(error)[mesh.triangles], 0.0)
    counts = finite.sum(axis=1)
    triangle_error = np.zeros(mesh.triangle_count, dtype=np.float64)
    np.divide(
        values.sum(axis=1), counts, out=triangle_error, where=counts > 0
    )
    return loss, triangle_error


def _domain_and_extraction(
    resolution: int,
    time_value: float = 0.0,
    manifold: str = "ripple",
):
    geometry_device = AbstractTensor._preferred_device or "cpu"
    geometry_precision = AbstractTensor.tensor(
        [0.0], dtype="float64", device=geometry_device
    ).get_dtype()
    domain = GridDomain.generate_grid_domain(
        "rectangular",
        N_u=resolution + 1,
        N_v=resolution + 1,
        N_w=resolution + 1,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        device=geometry_device,
        precision=geometry_precision,
        defer_resolution=True,
    )
    identity = compile_grid_domain(domain)
    expanded = _host(manifold_embedding(
        AbstractTensor.tensor(identity.parametric.reshape(-1, 3)),
        time_value,
        "ripple",
    ))
    compiled = DomainTetrahedra(
        identity.parametric,
        expanded[:, :3].reshape(identity.parametric.shape),
    )
    phase = 2.0 * np.pi * float(time_value)

    def time_surface_field(points):
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        target = 0.5 + 0.12 * (2.0 * np.pi * x + phase).sin() * (
            2.0 * np.pi * y
        ).cos()
        return z - target

    extraction = extract_isosurface(
        compiled.embedded,
        time_surface_field,
        parametric_tetrahedra=compiled.parametric,
        expanded_embedding=lambda points: manifold_embedding(
            AbstractTensor.tensor(points), time_value, manifold
        ),
    )
    return domain, extraction


def publish_surface_spline(samples) -> tuple[PublishedSurfaceSpline, int]:
    """Publish solely from values carried across the YoungMan boundary."""
    if samples.expanded_points is None:
        raise ValueError("YoungMan samples do not contain expanded geometry")
    validate_single_valued_chart(
        samples.parametric_points, intrinsic_axes=(0, 1), tolerance=1e-7
    )
    source_controls = np.asarray(samples.expanded_points, dtype=np.float64)
    parameter_batches = []
    value_batches = []
    for rows in np.array_split(np.arange(samples.sample_count), 12):
        parameter_batches.append(samples.parametric_points[rows])
        value_batches.append(source_controls[rows])
    parameters = np.concatenate(parameter_batches)
    values = np.concatenate(value_batches)
    model = AbstractKernelInterpolator.fit(
        parameters, values, intrinsic_axes=(0, 1)
    )
    return PublishedSurfaceSpline(model), len(parameters)


def build_blackbox_roundtrip(
    youngman_resolution: int = 7,
    position_tolerance: float = 1e-6,
    tangent_tolerance: float = 6e-1,
    *,
    max_rounds: int = 14,
    max_triangles: int = 250_000,
    time_value: float = 0.0,
    manifold: str = "ripple",
    train_network: bool = True,
    training_epochs: int = 80,
    training_target: str = "laplace",
    training_scope: str = "all",
    alpha_quantile: float = 0.85,
    training_examples: int = 9,
    spring_strength: float = 0.02,
    max_hinge_angle: float = 0.35,
    training_dtype: str = "float64",
    training_backend: str | None = None,
    training_device: str | None = None,
    training_seed: int = 1729,
) -> BlackBoxRoundTrip:
    """Build every stage while enforcing the spline/triangulator black box."""
    profile_rows = []
    if not 0.5 <= alpha_quantile < 1.0:
        raise ValueError("alpha_quantile must be in [0.5, 1.0)")
    if training_examples < 1 or spring_strength < 0.0:
        raise ValueError("invalid training corpus parameters")

    def finish_stage(name, started):
        profile_rows.append({
            "stage": name,
            "elapsed_sec": perf_counter() - started,
        })

    total_started = perf_counter()
    started = perf_counter()
    _, extraction = _domain_and_extraction(
        youngman_resolution, time_value, manifold
    )
    finish_stage("youngman_extract", started)
    samples = extraction.solver_samples
    assert samples is not None and samples.parametric_points is not None

    started = perf_counter()
    spline_surface, control_point_count = publish_surface_spline(samples)
    finish_stage("fifo_spline_fit", started)

    triangulator = AdaptiveSurfaceTriangulator(
        spline_surface,
        jacobian=spline_surface.jacobian,
        tolerance=TriangulationTolerance(
            position=position_tolerance,
            tangent=tangent_tolerance,
            hinge_angle=max_hinge_angle,
            max_rounds=max_rounds,
            max_triangles=max_triangles,
        ),
        initial_resolution=(4, 4),
    )
    started = perf_counter()
    mesh = triangulator.triangulate()
    finish_stage("adaptive_triangulation", started)

    trained = None
    training_features = None
    training_errors = None
    training_scale = position_tolerance
    pilot_mesh = mesh
    pilot_triangle_count = mesh.triangle_count
    pilot_laplace_loss = np.nan
    guided_laplace_loss = np.nan
    guided_triangle_count = pilot_triangle_count
    alpha_improved_objective = False
    guided_converged = False
    alpha_deployed = False
    alpha_inference_sec = 0.0
    deployment_reason = "model_not_accepted"
    laplace_targets = {"laplace", "discretization", "reconstruction"}
    if train_network and training_target in {"position", *laplace_targets}:
        started = perf_counter()
        if training_target == "position":
            feature_parts = []
            error_parts = []
            for certificate in mesh.certificate_history:
                features = triangle_refinement_features(
                    certificate.parameters, certificate.triangles
                )
                feature_parts.append(features)
                error_parts.append(certificate.position_error)
            training_features = np.concatenate(feature_parts)
            training_errors = np.concatenate(error_parts)
        else:
            corpus_features = []
            corpus_errors = []
            corpus_groups = []
            corpus_springs = []
            row_offset = 0
            candidates = [
                (name, phase)
                for phase in np.linspace(0.0, 1.0, 4, endpoint=False)
                for name in ("ripple", "banana", "saddle", "twisted_ribbon")
                if (name, float(phase)) != (manifold, float(time_value))
            ][:max(0, training_examples - 1)]
            cases = []
            for name, phase in candidates:
                _, case_extraction = _domain_and_extraction(
                    min(youngman_resolution, 3), float(phase), name
                )
                case_surface, _ = publish_surface_spline(
                    case_extraction.solver_samples
                )
                case_mesh = AdaptiveSurfaceTriangulator(
                    case_surface,
                    jacobian=case_surface.jacobian,
                    tolerance=TriangulationTolerance(
                        position=max(position_tolerance, 2e-2),
                        tangent=tangent_tolerance,
                        hinge_angle=max_hinge_angle,
                        max_rounds=min(max_rounds, 8),
                        max_triangles=min(max_triangles, 12_000),
                    ),
                    initial_resolution=(4, 4),
                ).triangulate()
                cases.append((
                    case_mesh, case_surface, float(phase), name
                ))
            # The requested target is the final group: labels are used only
            # for held-out validation and final deployment acceptance.
            cases.append((mesh, spline_surface, time_value, manifold))
            for group, (
                case_mesh, case_surface, phase, name
            ) in enumerate(cases):
                loss, errors = _mesh_laplace_objective(
                    case_mesh,
                    phase,
                    name,
                    spline_surface=case_surface,
                    component=(
                        "total"
                        if training_target == "laplace"
                        else training_target
                    ),
                )
                if group == len(cases) - 1:
                    pilot_laplace_loss, _ = _mesh_laplace_objective(
                        case_mesh,
                        phase,
                        name,
                        spline_surface=case_surface,
                        component="total",
                    )
                features = triangle_refinement_features(
                    case_mesh.parameters,
                    case_mesh.triangles,
                    case_mesh.embedded,
                )
                springs = triangle_spring_edges(case_mesh.triangles)
                if training_scope == "interior":
                    triangle_uv = case_mesh.parameters[
                        case_mesh.triangles
                    ]
                    keep = ~(
                        np.isclose(triangle_uv, 0.0)
                        | np.isclose(triangle_uv, 1.0)
                    ).any(axis=(1, 2))
                    remap = np.full(len(features), -1, dtype=np.int64)
                    remap[keep] = np.arange(int(keep.sum()))
                    spring_keep = (
                        (remap[springs[:, 0]] >= 0)
                        & (remap[springs[:, 1]] >= 0)
                    )
                    springs = remap[springs[spring_keep]]
                    features = features[keep]
                    errors = errors[keep]
                corpus_features.append(features)
                corpus_errors.append(errors)
                corpus_groups.append(
                    np.full(len(features), group, dtype=np.int64)
                )
                corpus_springs.append(springs + row_offset)
                row_offset += len(features)
            training_features = np.concatenate(corpus_features)
            training_errors = np.concatenate(corpus_errors)
            training_groups = np.concatenate(corpus_groups)
            training_springs = np.concatenate(corpus_springs)
            training_scale = max(float(np.median(training_errors)), 1e-12)
        if training_scope not in {"all", "interior"}:
            raise ValueError(f"unknown training scope: {training_scope}")
        if len(training_features) > 8192:
            selected = np.linspace(
                0, len(training_features) - 1, 8192, dtype=np.int64
            )
            if training_target in laplace_targets:
                remap = np.full(len(training_features), -1, dtype=np.int64)
                remap[selected] = np.arange(len(selected))
                spring_keep = (
                    (remap[training_springs[:, 0]] >= 0)
                    & (remap[training_springs[:, 1]] >= 0)
                )
                training_springs = remap[training_springs[spring_keep]]
                training_groups = training_groups[selected]
            training_features = training_features[selected]
            training_errors = training_errors[selected]
        selected_training_backend = (
            training_backend
            or AbstractTensor._preferred_backend
            or "numpy"
        )
        selected_training_device = (
            training_device
            if training_device is not None
            else AbstractTensor._preferred_device
        )
        finish_stage("training_corpus", started)
        with AbstractTensor.use_backend(
            selected_training_backend, selected_training_device
        ):
            trained = train_refinement_predictor(
                training_features,
                training_errors,
                epsilon=training_scale,
                epochs=training_epochs,
                group_ids=(
                training_groups if training_target in laplace_targets else None
                ),
                spring_edges=(
                training_springs if training_target in laplace_targets else None
                ),
                spring_strength=(
                    spring_strength if training_target in laplace_targets else 0.0
                ),
                tensor_dtype=training_dtype,
                seed=training_seed,
            )
        profile_rows.extend((
            {
                "stage": "training_tensor_setup",
                "elapsed_sec": trained.tensor_setup_sec,
            },
            {
                "stage": "abstract_nn_optimization",
                "elapsed_sec": trained.optimization_sec,
            },
            {
                "stage": "training_validation_inference",
                "elapsed_sec": trained.inference_sec,
            },
        ))

        if trained.accepted:
            def learned_alpha(parameters, triangles):
                nonlocal alpha_inference_sec
                inference_started = perf_counter()
                embedded = _host(spline_surface(parameters))
                pressure = trained.predict_alpha(
                    triangle_refinement_features(
                        parameters, triangles, embedded
                    )
                )
                threshold = max(
                    float(np.quantile(pressure, alpha_quantile)), 1e-12
                )
                result = pressure / threshold
                alpha_inference_sec += perf_counter() - inference_started
                return result

            guided = AdaptiveSurfaceTriangulator(
                spline_surface,
                jacobian=spline_surface.jacobian,
                tolerance=triangulator.tolerance,
                initial_resolution=triangulator.initial_resolution,
                batch_size=triangulator.batch_size,
                alpha_map=learned_alpha,
            )
            started = perf_counter()
            mesh = guided.triangulate()
            guided_converged = mesh.converged
            guided_triangle_count = mesh.triangle_count
            finish_stage("alpha_guided_triangulation", started)
            profile_rows.append({
                "stage": "alpha_inference_in_guided",
                "elapsed_sec": alpha_inference_sec,
            })
            if training_target in laplace_targets:
                started = perf_counter()
                guided_laplace_loss, _ = _mesh_laplace_objective(
                    mesh, time_value, manifold
                )
                finish_stage("guided_laplace_certification", started)
                alpha_improved_objective = (
                    guided_laplace_loss < pilot_laplace_loss
                )
                alpha_deployed, deployment_reason = deployment_decision(
                    model_accepted=trained.accepted,
                    guided_converged=guided_converged,
                    objective_improved=alpha_improved_objective,
                )
                if not alpha_deployed:
                    mesh = pilot_mesh
            else:
                alpha_deployed, deployment_reason = deployment_decision(
                    model_accepted=trained.accepted,
                    guided_converged=guided_converged,
                    objective_improved=True,
                )
                if not alpha_deployed:
                    mesh = pilot_mesh

    started = perf_counter()
    uv = mesh.parameters
    source_values = source_surface(uv, time_value, manifold)
    spline_values = mesh.embedded
    source_jacobian = source_surface_jacobian(uv, time_value, manifold)
    spline_jacobian_tensor = spline_surface.jacobian(uv)
    spline_jacobian_values = _host(spline_jacobian_tensor)
    source_metric = _metric(source_jacobian)
    spline_metric = _metric(spline_jacobian_values)
    source_continuous_laplace = continuous_surface_laplace(
        uv,
        lambda query: _metric(
            source_surface_jacobian(query, time_value, manifold)
        ),
    )
    spline_continuous_laplace = continuous_surface_laplace(
        uv, lambda query: _metric(spline_surface.jacobian(query))
    )
    finish_stage("continuous_reference", started)

    started = perf_counter()
    geometry_transform = TriangulatedSurfaceTransform.from_mesh(
        mesh.parameters, mesh.embedded, mesh.triangles
    )
    probe = probe_values(uv)
    mesh_result = geometry_transform.laplace(probe)
    abstract_laplace_tensor = abstract_mesh_laplace(
        AbstractTensor.tensor(mesh.embedded, dtype="float64"),
        mesh.triangles,
        AbstractTensor.tensor(probe, dtype="float64"),
    )
    mesh_laplace = _host(abstract_laplace_tensor)
    abstract_laplace_parity = np.nanmax(
        np.abs(mesh_laplace - mesh_result.laplacian)
    )
    finish_stage("mesh_transform_laplace", started)
    boundary = mesh_result.geometry.boundary_vertex_mask
    interior = (
        ~boundary
        & ~mesh_result.geometry.invalid_vertex_mask
    )

    started = perf_counter()
    youngman_error = np.linalg.norm(
        samples.embedded_points
        - detailed_embedding(samples.parametric_points)[:, :3],
        axis=1,
    )
    spline_error = np.linalg.norm(spline_values - source_values, axis=1)
    metric_error = np.linalg.norm(spline_metric - source_metric, axis=(1, 2))
    continuous_laplace_error = (
        spline_continuous_laplace - source_continuous_laplace
    )
    mesh_discretization_error = mesh_laplace - spline_continuous_laplace
    mesh_laplace_error = mesh_laplace - source_continuous_laplace

    training_rows = []
    if train_network and trained is None:
        started = perf_counter()
        all_training_features = triangle_refinement_features(
            mesh.parameters, mesh.triangles
        )
        target_fields = {
            "position": mesh.position_error,
            "spline_position": spline_error[mesh.triangles].mean(axis=1),
            "metric": metric_error[mesh.triangles].mean(axis=1),
            "laplace": np.nanmean(
                np.abs(mesh_laplace_error)[mesh.triangles], axis=1
            ),
        }
        if training_target not in target_fields:
            raise ValueError(f"unknown training target: {training_target}")
        training_errors = target_fields[training_target]
        training_mask = np.isfinite(training_errors)
        if training_scope == "interior":
            training_mask &= ~boundary[mesh.triangles].any(axis=1)
        elif training_scope != "all":
            raise ValueError(f"unknown training scope: {training_scope}")
        training_features = all_training_features[training_mask]
        training_errors = training_errors[training_mask]
        if len(training_features) > 2048:
            selected = np.linspace(
                0, len(training_features) - 1, 2048, dtype=np.int64
            )
            training_features = training_features[selected]
            training_errors = training_errors[selected]
        training_scale = (
            position_tolerance
            if training_target == "position"
            else max(float(np.median(training_errors)), 1e-12)
        )
        trained = train_refinement_predictor(
            training_features,
            training_errors,
            epsilon=training_scale,
            epochs=training_epochs,
            seed=training_seed,
        )
        finish_stage("abstract_nn_training", started)
    if train_network:
        training_rows.append({
            "engine": (
                "abstract_nn.Sequential+FusedProgram+"
                "tape_reverse_mode+AutogradProcess"
            ),
            "backend": trained.tensor_backend or "auto",
            "device": trained.tensor_device or "default",
            "dtype": trained.tensor_dtype,
            "target": training_target,
            "scope": training_scope,
            "target_scale": training_scale,
            "epochs": trained.epochs,
            "seed": trained.seed,
            "samples": len(training_features),
            "initial_loss": trained.initial_loss,
            "final_loss": trained.final_loss,
            "loss_ratio": trained.final_loss / max(
                trained.initial_loss, np.finfo(float).tiny
            ),
            "forward_nodes": trained.forward_nodes,
            "backward_nodes": trained.backward_nodes,
            "concurrent_forward_width": trained.concurrent_forward_width,
            "validation_loss": trained.validation_loss,
            "baseline_validation_loss": trained.baseline_validation_loss,
            "validation_correlation": trained.validation_correlation,
            "accepted": trained.accepted,
            "guided_converged": guided_converged,
            "pilot_triangles": pilot_triangle_count,
            "guided_triangles": guided_triangle_count,
            "alpha_applied": alpha_deployed,
            "deployment_reason": deployment_reason,
            "pilot_laplace_loss": pilot_laplace_loss,
            "guided_laplace_loss": guided_laplace_loss,
            "alpha_improved_objective": alpha_improved_objective,
        })

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values)))) if len(values) else np.nan

    vertex_weights = mesh_result.geometry.lumped_vertex_areas

    def weighted_rms(values, mask=None):
        values = np.asarray(values, dtype=np.float64)
        selected = np.ones(len(values), dtype=bool) if mask is None else mask
        selected &= np.isfinite(values)
        weights = vertex_weights[selected]
        if not len(weights) or weights.sum() <= 0.0:
            return np.nan
        return float(np.sqrt(np.sum(weights * values[selected] ** 2) / weights.sum()))

    summary = pd.DataFrame([{
        "tensor_backend": AbstractTensor._preferred_backend or "auto",
        "tensor_device": AbstractTensor._preferred_device or "default",
        "time_value": time_value,
        "manifold": manifold,
        "target_epsilon": position_tolerance,
        "youngman_resolution": youngman_resolution,
        "youngman_samples": samples.sample_count,
        "spline_controls": control_point_count,
        "spline_embedding_dimension": spline_surface.model.embedding_dimension,
        "triangulation_generation": mesh.generation,
        "mesh_vertices": len(mesh.parameters),
        "mesh_triangles": mesh.triangle_count,
        "mesh_converged": mesh.converged,
        "mesh_surface_sample_rows": mesh.surface_sample_count,
        "mesh_jacobian_sample_rows": mesh.jacobian_sample_count,
        "youngman_error_rms": rms(youngman_error),
        "spline_position_error_area_rms": weighted_rms(spline_error),
        "spline_metric_error_area_rms": weighted_rms(metric_error),
        "triangulator_max_chord_error": float(mesh.position_error.max()),
        "epsilon_ratio": float(mesh.position_error.max() / position_tolerance),
        "epsilon_achieved": bool(
            float(mesh.position_error.max()) <= position_tolerance
        ),
        "triangulator_max_tangent_error": (
            float(mesh.tangent_error.max()) if mesh.tangent_error is not None else np.nan
        ),
        "continuous_spline_laplace_error_area_rms_interior": weighted_rms(
            continuous_laplace_error, interior
        ),
        "mesh_discretization_error_area_rms_interior": weighted_rms(
            mesh_discretization_error, interior
        ),
        "source_laplace_area_rms_interior": weighted_rms(
            source_continuous_laplace, interior
        ),
        "mesh_laplace_error_area_rms_interior": weighted_rms(
            mesh_laplace_error, interior
        ),
        "degenerate_mesh_vertices": int(
            mesh_result.geometry.degenerate_vertex_mask.sum()
        ),
        "nonmanifold_mesh_edges": int(
            mesh_result.geometry.nonmanifold_edge_mask.sum()
        ),
        "abstract_laplace_parity_max": abstract_laplace_parity,
        "abstract_nn_trained": bool(training_rows),
        "abstract_nn_loss_ratio": (
            training_rows[0]["loss_ratio"] if training_rows else np.nan
        ),
        "abstract_nn_accepted": (
            training_rows[0]["accepted"] if training_rows else False
        ),
        "abstract_nn_deployed": (
            training_rows[0]["alpha_applied"] if training_rows else False
        ),
    }])
    triangle_position = spline_error[mesh.triangles].mean(axis=1)
    triangle_chord = mesh.position_error
    triangle_metric = metric_error[mesh.triangles].mean(axis=1)
    def triangle_interior_mean(values):
        gathered = np.where(interior, values, np.nan)[mesh.triangles]
        count = np.isfinite(gathered).sum(axis=1)
        total = np.nansum(gathered, axis=1)
        return np.divide(
            total, count, out=np.full(len(count), np.nan), where=count > 0
        )

    triangle_laplace = triangle_interior_mean(mesh_laplace_error)
    triangle_discretization = triangle_interior_mean(
        mesh_discretization_error
    )
    triangle_report = pd.DataFrame({
        "triangle": np.arange(mesh.triangle_count),
        "spline_position_error": triangle_position,
        "triangulation_chord_error": triangle_chord,
        "spline_metric_error": triangle_metric,
        "mesh_laplace_error": triangle_laplace,
        "mesh_discretization_error": triangle_discretization,
        "touches_boundary": boundary[mesh.triangles].any(axis=1),
    })
    finish_stage("error_reporting", started)
    profile_rows.append({
        "stage": "total",
        "elapsed_sec": perf_counter() - total_started,
    })
    return BlackBoxRoundTrip(
        summary,
        triangle_report,
        mesh,
        geometry_transform,
        pd.DataFrame(profile_rows),
        pd.DataFrame(training_rows),
    )


def _load_pluck_viewer():
    root = Path(__file__).resolve().parents[5]
    pluck = root / "spectral-analyzer"
    if str(pluck) not in sys.path:
        sys.path.insert(0, str(pluck))
    import ordinary_gl_mesh_viewer
    return ordinary_gl_mesh_viewer


def _triangle_field(result, name: str) -> np.ndarray | None:
    if name == "geometry":
        return None
    fields = {
        "youngman": np.full(
            result.mesh.triangle_count,
            result.summary.loc[0, "youngman_error_rms"],
        ),
        "spline": result.triangles["spline_position_error"].to_numpy(),
        "triangulation": result.triangles[
            "triangulation_chord_error"
        ].to_numpy(),
        "metric": result.triangles["spline_metric_error"].to_numpy(),
        "laplace": result.triangles["mesh_laplace_error"].to_numpy(),
    }
    return fields[name]


def _profile_mapping(result: BlackBoxRoundTrip) -> dict[str, float]:
    return dict(zip(result.profile["stage"], result.profile["elapsed_sec"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--youngman-resolution", type=int, default=7)
    parser.add_argument(
        "--target-epsilon",
        type=float,
        default=1e-6,
        help="target maximum positional certificate (default: 1e-6)",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--tangent-tolerance", type=float, default=6e-1)
    parser.add_argument("--max-rounds", type=int, default=14)
    parser.add_argument("--max-triangles", type=int, default=250_000)
    parser.add_argument("--time-value", type=float, default=0.0)
    parser.add_argument(
        "--manifold",
        choices=("ripple", "banana", "saddle", "twisted_ribbon"),
        default="banana",
    )
    parser.add_argument("--animation", type=Path)
    parser.add_argument("--animation-frames", type=int, default=8)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-solves", type=int)
    parser.add_argument("--live-period", type=float, default=8.0)
    parser.add_argument("--live-max-frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument(
        "--tensor-backend", choices=("numpy", "torch", "c"), default="numpy"
    )
    parser.add_argument(
        "--tensor-device",
        default=None,
        help="backend device such as cpu, cuda, or cuda:1",
    )
    parser.add_argument(
        "--training-dtype",
        choices=("float32", "float64"),
        default=None,
    )
    parser.add_argument(
        "--training-backend",
        choices=("numpy", "torch", "c"),
        default=None,
        help="override the geometry backend for neural training only",
    )
    parser.add_argument(
        "--training-device",
        default=None,
        help="training-only device such as cuda or cuda:1",
    )
    parser.add_argument("--training-epochs", type=int, default=80)
    parser.add_argument("--training-seed", type=int, default=1729)
    parser.add_argument(
        "--training-target",
        choices=(
            "position",
            "spline_position",
            "metric",
            "laplace",
            "discretization",
            "reconstruction",
        ),
        default="laplace",
    )
    parser.add_argument(
        "--training-scope", choices=("all", "interior"), default="all"
    )
    parser.add_argument("--alpha-quantile", type=float, default=0.85)
    parser.add_argument("--training-examples", type=int, default=9)
    parser.add_argument("--spring-strength", type=float, default=0.02)
    parser.add_argument(
        "--max-hinge-angle",
        type=float,
        default=0.35,
        help="maximum principal angle in radians across connected faces",
    )
    parser.add_argument("--render-image", type=Path)
    parser.add_argument(
        "--error-field",
        choices=(
            "geometry", "youngman", "spline", "triangulation", "metric",
            "laplace",
        ),
        default="geometry",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="emit diagnostics even if a triangulation tolerance or budget fails",
    )
    args = parser.parse_args()
    AbstractTensor.set_default_backend(
        args.tensor_backend, args.tensor_device
    )
    effective_training_backend = (
        args.training_backend or args.tensor_backend
    )
    effective_training_device = (
        args.training_device
        if args.training_device is not None
        else args.tensor_device
    )
    training_dtype = args.training_dtype or (
        "float32"
        if effective_training_backend == "torch"
        and effective_training_device
        and effective_training_device.startswith("cuda")
        else "float64"
    )
    target_epsilon = (
        args.target_epsilon
        if args.position_tolerance is None
        else args.position_tolerance
    )
    result = build_blackbox_roundtrip(
        args.youngman_resolution,
        target_epsilon,
        args.tangent_tolerance,
        max_rounds=args.max_rounds,
        max_triangles=args.max_triangles,
        time_value=args.time_value,
        manifold=args.manifold,
        train_network=not args.no_train,
        training_epochs=args.training_epochs,
        training_target=args.training_target,
        training_scope=args.training_scope,
        alpha_quantile=args.alpha_quantile,
        training_examples=args.training_examples,
        spring_strength=args.spring_strength,
        max_hinge_angle=args.max_hinge_angle,
        training_dtype=training_dtype,
        training_backend=args.training_backend,
        training_device=args.training_device,
        training_seed=args.training_seed,
    )
    print("\nBLACK-BOX ROUND TRIP\n", result.summary.to_string(index=False))
    print("\nPROFILE\n", result.profile.to_string(index=False))
    if len(result.training):
        print("\nABSTRACT NN TRAINING\n", result.training.to_string(index=False))
    print("\nTRIANGLE CERTIFICATES\n", result.triangles.head(12).to_string(index=False))
    if not result.mesh.converged and not args.allow_unconverged:
        raise RuntimeError(
            f"triangulation did not converge: {result.mesh.stopped_reason}; "
            "use --allow-unconverged for failure diagnostics"
        )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(args.output_dir / "summary.csv", index=False)
        result.triangles.to_csv(args.output_dir / "triangles.csv", index=False)
        result.profile.to_csv(args.output_dir / "profile.csv", index=False)
        result.training.to_csv(args.output_dir / "training.csv", index=False)
    if args.render_image:
        viewer = _load_pluck_viewer()
        profile = _profile_mapping(result)
        panel = viewer.rolling_profile_lines(
            profile, (profile,), time_value=args.time_value
        )
        output = viewer.render_triangle_mesh_image(
            result.mesh.triangle_soup,
            args.render_image,
            triangle_values=_triangle_field(result, args.error_field),
            value_label=f"{args.error_field} stage error",
            title="Black-box geometry round trip",
            side_panel_lines=panel,
        )
        print(f"\nHEADLESS IMAGE\n {output}")
    if args.animation:
        if args.animation_frames < 2:
            raise ValueError("--animation-frames must be at least 2")
        from PIL import Image

        viewer = _load_pluck_viewer()
        frame_root = args.animation.with_suffix("")
        frame_root.mkdir(parents=True, exist_ok=True)
        history = []
        images = []
        for frame, time_value in enumerate(
            np.linspace(0.0, 1.0, args.animation_frames, endpoint=False)
        ):
            animated = result if frame == 0 and args.time_value == 0.0 else (
                build_blackbox_roundtrip(
                    args.youngman_resolution,
                    target_epsilon,
                    args.tangent_tolerance,
                    max_rounds=args.max_rounds,
                    max_triangles=args.max_triangles,
                    time_value=float(time_value),
                    manifold=args.manifold,
                    train_network=not args.no_train,
                    training_epochs=args.training_epochs,
                    training_target=args.training_target,
                    training_scope=args.training_scope,
                    alpha_quantile=args.alpha_quantile,
                    training_examples=args.training_examples,
                    spring_strength=args.spring_strength,
                    max_hinge_angle=args.max_hinge_angle,
                    training_dtype=training_dtype,
                    training_backend=args.training_backend,
                    training_device=args.training_device,
                    training_seed=args.training_seed,
                )
            )
            if not animated.mesh.converged and not args.allow_unconverged:
                raise RuntimeError(
                    f"animation frame {frame} at t={time_value:.4f} did not "
                    f"converge: {animated.mesh.stopped_reason}"
                )
            profile = _profile_mapping(animated)
            history.append(profile)
            panel = viewer.rolling_profile_lines(
                profile, history, time_value=float(time_value)
            )
            frame_path = frame_root / f"frame_{frame:03d}.png"
            viewer.render_triangle_mesh_image(
                animated.mesh.triangle_soup,
                frame_path,
                triangle_values=_triangle_field(animated, args.error_field),
                value_label=f"{args.error_field} stage error",
                title="Time-varying black-box solve",
                side_panel_lines=panel,
            )
            images.append(Image.open(frame_path).convert("RGB"))
        args.animation.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            args.animation,
            save_all=True,
            append_images=images[1:],
            duration=900,
            loop=0,
        )
        for image in images:
            image.close()
        print(f"\nPROFILED ANIMATION\n {args.animation.resolve()}")
    if args.live:
        viewer = _load_pluck_viewer()
        history = []

        def solve_live_frame(index, time_value):
            solved = build_blackbox_roundtrip(
                args.youngman_resolution,
                target_epsilon,
                args.tangent_tolerance,
                max_rounds=args.max_rounds,
                max_triangles=args.max_triangles,
                time_value=time_value,
                manifold=args.manifold,
                train_network=not args.no_train,
                training_epochs=args.training_epochs,
                training_target=args.training_target,
                training_scope=args.training_scope,
                alpha_quantile=args.alpha_quantile,
                training_examples=args.training_examples,
                spring_strength=args.spring_strength,
                max_hinge_angle=args.max_hinge_angle,
                training_dtype=training_dtype,
                training_backend=args.training_backend,
                training_device=args.training_device,
                training_seed=args.training_seed,
            )
            profile = _profile_mapping(solved)
            history.append(profile)
            panel = viewer.rolling_profile_lines(
                profile, history, time_value=time_value
            )
            training = (
                solved.training.iloc[0] if len(solved.training) else None
            )
            panel.extend((
                "",
                "benchmark          TRAIN -> HELDOUT EVAL",
                f"solve index        {index:8d}",
                f"manifold           {args.manifold:>8}",
                f"corpus examples    {args.training_examples:8d}",
                f"held-out phase     {time_value:8.4f}",
                f"certified          {str(solved.mesh.converged):>8}",
                f"target epsilon     {target_epsilon:8.2e}",
                f"epsilon ratio      {solved.summary.loc[0, 'epsilon_ratio']:8.3f}",
                f"vertices           {len(solved.mesh.parameters):8d}",
                f"triangles          {solved.mesh.triangle_count:8d}",
                f"NN loss ratio      "
                f"{solved.summary.loc[0, 'abstract_nn_loss_ratio']:8.3g}",
                f"NN accepted        "
                f"{str(solved.summary.loc[0, 'abstract_nn_accepted']):>8}",
                (
                    f"pilot LB RMS       {training['pilot_laplace_loss']:8.3g}"
                    if training is not None else "pilot LB RMS            n/a"
                ),
                (
                    f"guided LB RMS      {training['guided_laplace_loss']:8.3g}"
                    if training is not None else "guided LB RMS           n/a"
                ),
                (
                    f"deployed           "
                    f"{str(training['alpha_applied']):>8}"
                    if training is not None else "deployed                 n/a"
                ),
            ))
            return viewer.LiveMeshFrame(
                solved.mesh.triangle_soup,
                _triangle_field(solved, args.error_field),
                panel,
                time_value,
            )

        viewer.view_profiled_triangle_mesh_stream(
            solve_live_frame,
            period_sec=args.live_period,
            max_solves=args.live_solves,
            max_frames=args.live_max_frames,
        )


if __name__ == "__main__":
    main()
