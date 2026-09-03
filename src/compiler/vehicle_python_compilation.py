"""Compile the shared vectorized vehicle Python graph through ProcessGraph.

This is the active replacement boundary for the former handwritten native C
assembly.  Existing symbolic equations are linked as ProcessGraph functions;
the adapters below only pack/unpack their named tensor ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from functools import reduce
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import sympy

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics
from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot

from .abstract_ui_vehicles import WHEEL_NAMES, compile_symbolic_vehicle_physics
from .vehicle_balloon_tire import (
    balloon_tire_linked_process_graphs,
    balloon_tire_python_bindings,
    compile_balloon_bead_implicit_step_ssa,
    compile_balloon_gas_ssa,
    compile_balloon_membrane_face_ssa,
)
from .vehicle_balloon_tire_program import (
    BALLOON_TIRE_VECTOR_SOURCE,
    balloon_tire_python_program,
)
from .vehicle_mechanical_material import compile_vehicle_member_material_ssa
from .vehicle_native_graph_program import (
    BATCH_CAPACITY,
    RIG_POINT_COUNT,
    VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE,
    VehicleGraphConstants,
    VehicleNativeGraphPythonProgram,
    vehicle_native_graph_python_program,
)


FIXTURE_CORNERS = ("front_left", "front_right", "rear_left", "rear_right")


def _runtime_maximum(*values: Any) -> Any:
    def combine(left, right):
        if hasattr(left, "maximum"):
            return left.maximum(right)
        if hasattr(right, "maximum"):
            return right.maximum(left)
        return max(left, right)
    return reduce(combine, values)


def _runtime_minimum(*values: Any) -> Any:
    def combine(left, right):
        if hasattr(left, "minimum"):
            return left.minimum(right)
        if hasattr(right, "minimum"):
            return right.minimum(left)
        return min(left, right)
    return reduce(combine, values)


def _runtime_unary(value: Any, method: str, scalar) -> Any:
    operation = getattr(value, method, None)
    return operation() if operation is not None else scalar(value)


@lru_cache(maxsize=2)
def _vehicle_python_runtime_bindings_cached(
    include_configured_vehicle: bool,
) -> Mapping[str, Any]:
    """Lambdify the exact symbolic callees used by the authored graph.

    This is an eager execution boundary only.  Compiler lowering continues to
    receive the retained process graphs and never receives these callables.
    """

    compilations = {
        "vehicle_member_material_step": compile_vehicle_member_material_ssa(),
    }
    if include_configured_vehicle:
        compilations["abstract_ui_vehicle_step"] = (
            compile_symbolic_vehicle_physics())
    bindings = balloon_tire_python_bindings()
    modules = [{
        "Max": _runtime_maximum,
        "Min": _runtime_minimum,
        "sqrt": lambda value: _runtime_unary(value, "sqrt", math.sqrt),
        "sin": lambda value: _runtime_unary(value, "sin", math.sin),
        "cos": lambda value: _runtime_unary(value, "cos", math.cos),
        "Abs": lambda value: _runtime_unary(value, "abs", abs),
        "tanh": lambda value: _runtime_unary(value, "tanh", math.tanh),
        "exp": lambda value: _runtime_unary(value, "exp", math.exp),
    }]
    for name, compilation in compilations.items():
        metadata = compilation.function.metadata
        equations_by_name = {
            str(equation.lhs): equation.rhs
            for equation in compilation.equations
        }
        bindings[name] = sympy.lambdify(
            tuple(sympy.Symbol(argument) for argument in metadata["argument_names"]),
            tuple(equations_by_name[output] for output in metadata["output_names"]),
            modules=modules,
            cse=True,
        )
    return bindings


def vehicle_python_runtime_bindings(
    *, include_configured_vehicle: bool = True,
) -> dict[str, Any]:
    """Return eager bindings for executing the compiler-authored Python."""

    return dict(_vehicle_python_runtime_bindings_cached(
        include_configured_vehicle))


def _names(compilation: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    metadata = compilation.function.metadata
    return tuple(metadata["argument_names"]), tuple(metadata["output_names"])


def _adapter_source(vehicle: Any) -> str:
    vehicle_inputs, vehicle_outputs = _names(vehicle)
    vi = {name: index for index, name in enumerate(vehicle_inputs)}
    required_vehicle = {
        *(f"contact_normal_force_{corner}" for corner in FIXTURE_CORNERS),
        *(f"tire_reaction_torque_{corner}" for corner in FIXTURE_CORNERS),
        *(f"material_plastic_set_{corner}" for corner in FIXTURE_CORNERS),
        *(f"material_survival_{corner}" for corner in FIXTURE_CORNERS),
        *(f"contact_wrench_force_{axis}" for axis in "xyz"),
        *(f"contact_wrench_torque_{axis}" for axis in "xyz"),
    }
    if not required_vehicle <= vi.keys():
        raise RuntimeError("vehicle Python adapter missing " + repr(sorted(required_vehicle - vi.keys())))
    lines = [
        "def vehicle_physics_step_vector(vehicle_input, wheel_load, reaction_torque, total_force, total_torque, corner_plastic, corner_survival, structural_support_position):",
    ]
    for wheel, corner in enumerate(FIXTURE_CORNERS):
        lines.extend((
            f"    vehicle_input[:, {vi[f'contact_normal_force_{corner}']}] = wheel_load[:, {wheel}]",
            f"    vehicle_input[:, {vi[f'tire_reaction_torque_{corner}']}] = reaction_torque[:, {wheel}]",
            f"    vehicle_input[:, {vi[f'material_plastic_set_{corner}']}] = corner_plastic[:, {wheel}]",
            f"    vehicle_input[:, {vi[f'material_survival_{corner}']}] = corner_survival[:, {wheel}]",
        ))
    for axis, index in zip("xyz", range(3)):
        lines.extend((
            f"    vehicle_input[:, {vi[f'contact_wrench_force_{axis}']}] = total_force[:, {index}]",
            f"    vehicle_input[:, {vi[f'contact_wrench_torque_{axis}']}] = total_torque[:, {index}]",
        ))
    vehicle_arguments = ", ".join(
        f"vehicle_input[:, {vi[name]}]" for name in vehicle_inputs
    )
    lines.extend((
        f"    result = abstract_ui_vehicle_step({vehicle_arguments})",
        "    return AbstractTensor.stack(result, dim=1)",
        "",
    ))
    return "\n".join(lines)


def _machine_adapter_source(
    input_names: tuple[str, ...], output_names: tuple[str, ...],
) -> str:
    """Author the general structural-body component without loading a car.

    Optional suspension, steering and powertrain components remain separate
    reusable graph components.  This operator owns only rigid pose/inertia and
    the wrench delivered through the selected structural grasp coordinates.
    """

    vi = {name: index for index, name in enumerate(input_names)}
    required = {
        "dt", "gravity", "inverse_mass", "inverse_inertia_roll",
        "inverse_inertia_pitch", "inverse_inertia_yaw", "angular_damping",
        "position_x", "position_y", "position_z", "velocity_x",
        "velocity_y", "velocity_z", "roll", "pitch", "yaw",
        "roll_velocity", "pitch_velocity", "yaw_velocity",
    }
    if not required <= vi.keys():
        raise RuntimeError(
            "general machine body ABI missing " + repr(sorted(required - vi.keys())))
    lines = [
        "def vehicle_physics_step_vector(vehicle_input, wheel_load, reaction_torque, total_force, total_torque, corner_plastic, corner_survival, structural_support_position):",
        f"    dt = vehicle_input[:, {vi['dt']}]",
        "    support_position = structural_support_position.reshape((1, structural_support_position.shape[0], 3)) * AbstractTensor.ones_like(wheel_load).reshape((-1, wheel_load.shape[1], 1))",
        "    support_force = support_position * 0.0",
        "    support_force[:, :, 1] = wheel_load",
        "    body_force = total_force + support_force.sum(dim=1)",
        "    body_torque = total_torque + graph_cross(support_position, support_force).sum(dim=1)",
        f"    acceleration_x = body_force[:, 0] * vehicle_input[:, {vi['inverse_mass']}]",
        f"    acceleration_y = body_force[:, 1] * vehicle_input[:, {vi['inverse_mass']}] + vehicle_input[:, {vi['gravity']}]",
        f"    acceleration_z = body_force[:, 2] * vehicle_input[:, {vi['inverse_mass']}]",
        f"    velocity_x_next = vehicle_input[:, {vi['velocity_x']}] + dt * acceleration_x",
        f"    velocity_y_next = vehicle_input[:, {vi['velocity_y']}] + dt * acceleration_y",
        f"    velocity_z_next = vehicle_input[:, {vi['velocity_z']}] + dt * acceleration_z",
        f"    position_x_next = vehicle_input[:, {vi['position_x']}] + dt * velocity_x_next",
        f"    position_y_next = vehicle_input[:, {vi['position_y']}] + dt * velocity_y_next",
        f"    position_z_next = vehicle_input[:, {vi['position_z']}] + dt * velocity_z_next",
        f"    roll_velocity_next = (vehicle_input[:, {vi['roll_velocity']}] + dt * body_torque[:, 0] * vehicle_input[:, {vi['inverse_inertia_roll']}]) / (1.0 + dt * vehicle_input[:, {vi['angular_damping']}])",
        f"    pitch_velocity_next = (vehicle_input[:, {vi['pitch_velocity']}] + dt * body_torque[:, 1] * vehicle_input[:, {vi['inverse_inertia_pitch']}]) / (1.0 + dt * vehicle_input[:, {vi['angular_damping']}])",
        f"    yaw_velocity_next = (vehicle_input[:, {vi['yaw_velocity']}] + dt * body_torque[:, 2] * vehicle_input[:, {vi['inverse_inertia_yaw']}]) / (1.0 + dt * vehicle_input[:, {vi['angular_damping']}])",
        f"    roll_next = vehicle_input[:, {vi['roll']}] + dt * roll_velocity_next",
        f"    pitch_next = vehicle_input[:, {vi['pitch']}] + dt * pitch_velocity_next",
        f"    yaw_next = vehicle_input[:, {vi['yaw']}] + dt * yaw_velocity_next",
        f"    zero = vehicle_input[:, {vi['dt']}] * 0.0",
    ]
    calculated = {
        name: name for name in (
            "position_x_next", "position_y_next", "position_z_next",
            "velocity_x_next", "velocity_y_next", "velocity_z_next",
            "roll_next", "pitch_next", "yaw_next", "roll_velocity_next",
            "pitch_velocity_next", "yaw_velocity_next")
    }
    values = []
    for name in output_names:
        if name in calculated:
            values.append(calculated[name])
        elif name.endswith("_next") and name[:-5] in vi:
            base = name[:-5]
            if base.startswith("wheel_angle_"):
                omega = base.replace("wheel_angle_", "wheel_omega_", 1)
                values.append(
                    f"vehicle_input[:, {vi[base]}] + dt * vehicle_input[:, {vi[omega]}]"
                    if omega in vi else f"vehicle_input[:, {vi[base]}]")
            else:
                values.append(f"vehicle_input[:, {vi[base]}]")
        else:
            values.append("zero")
    lines.append("    return AbstractTensor.stack([")
    lines.extend(f"        {value}," for value in values)
    lines.extend(("    ], dim=1)", ""))
    return "\n".join(lines)


def _periodic_terrain_triangles() -> np.ndarray:
    def height(ix: int, iz: int) -> float:
        tau = 2.0 * math.pi
        u, v = tau * (ix % 8) / 8.0, tau * (iz % 8) / 8.0
        return 0.075 * math.sin(u) + 0.045 * math.sin(2.0 * v + 0.7) + 0.035 * math.sin(u + 1.5 * v)

    triangles = []
    for iz in range(8):
        for ix in range(8):
            a = (ix / 8.0, height(ix, iz), iz / 8.0)
            b = ((ix + 1) / 8.0, height(ix + 1, iz), iz / 8.0)
            c = (ix / 8.0, height(ix, iz + 1), (iz + 1) / 8.0)
            d = ((ix + 1) / 8.0, height(ix + 1, iz + 1), (iz + 1) / 8.0)
            triangles.extend(((a, b, c), (d, c, b)))
    return np.asarray(triangles, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VehiclePythonCompilationInputs:
    source: str
    entrypoint: str
    feeds: Mapping[str, Any]
    linked_process_graphs: Mapping[str, Any]

    def abstract_tensor_feeds(self) -> dict[str, Any]:
        """Materialize this one program layout for eager AbstractTensor use.

        Native repository ABIs store arenas as doubles. Gather indices and
        selectors recover their semantic dtype only at this host boundary;
        the authored source and its feed identities remain unchanged.
        """

        index_names = {
            "edge_nodes", "tire_wheel_input_indices", "tire_face_vertices",
        }
        mask_names = {
            "tire_bead_mask", "roller_anchor_valid", "tire_initialized",
            "tire_history_valid",
        }
        materialized: dict[str, Any] = {}
        for name, value in self.feeds.items():
            if not isinstance(value, np.ndarray):
                materialized[name] = value
            elif name in index_names:
                materialized[name] = value.astype(np.int64, copy=True)
            elif name in mask_names:
                materialized[name] = value.astype(bool, copy=True)
            else:
                materialized[name] = value.copy()
        return materialized


@dataclass(frozen=True, slots=True)
class VehiclePythonSSALowering:
    module: Any
    root_name: str
    outputs: Mapping[str, Any]
    exports: tuple[str, ...]


@dataclass
class BalloonTireManagedState:
    """Rollback-complete tire material consumed by the repository dt system."""

    inputs: np.ndarray
    state: np.ndarray
    output: np.ndarray
    wheel_input_indices: np.ndarray
    rest: np.ndarray
    face_vertices: np.ndarray
    face_rest: np.ndarray
    face_scatter: np.ndarray
    bending_incidence: np.ndarray
    bending_scatter: np.ndarray
    bending_weight: np.ndarray
    vertex_area: np.ndarray
    bead_mask: np.ndarray
    face_material: np.ndarray
    telemetry: np.ndarray
    displacement_criticality_m: float
    last_maximum_displacement_m: float = 0.0
    last_maximum_velocity_m_s: float = 0.0

    def copy_shallow(self):
        return (
            self.inputs.copy(),
            self.state.copy(),
            self.output.copy(),
            float(self.last_maximum_displacement_m),
            float(self.last_maximum_velocity_m_s),
        )

    def restore(self, snapshot) -> None:
        inputs, state, output, displacement, velocity = snapshot
        self.inputs[...] = inputs
        self.state[...] = state
        self.output[...] = output
        self.last_maximum_displacement_m = float(displacement)
        self.last_maximum_velocity_m_s = float(velocity)


BALLOON_TIRE_MANAGED_SOURCE = r'''
def balloon_tire_managed_advance(material, dt):
    material.telemetry[1] = material.telemetry[1] + 1.0
    material.telemetry[5] = min(material.telemetry[5], dt)
    material.telemetry[6] = max(material.telemetry[6], dt)
    material.telemetry[7] = dt
    previous_position = material.state[:, :, :, 0:3] + 0.0
    material.inputs[:, 0] = dt
    material.state, material.output = balloon_tire_vector_step(
        material.inputs, material.state, material.output,
        material.wheel_input_indices, material.rest,
        material.face_vertices, material.face_rest, material.face_scatter,
        material.bending_incidence, material.bending_scatter,
        material.bending_weight, material.vertex_area, material.bead_mask,
        material.face_material)
    position_delta = material.state[:, :, :, 0:3] - previous_position
    maximum_displacement = (
        (position_delta * position_delta).sum(dim=3) + 1.0e-30
    ).sqrt().max()
    velocity = material.state[:, :, :, 3:6]
    maximum_velocity = (
        (velocity * velocity).sum(dim=3) + 1.0e-30
    ).sqrt().max()
    finite = material.state.isfinite().all() and material.output.isfinite().all()
    if finite:
        material.telemetry[11] = max(
            material.telemetry[11], maximum_displacement
        )
        material.telemetry[14] = max(
            material.telemetry[14], maximum_velocity
        )
    else:
        material.telemetry[4] = material.telemetry[4] + 1.0
    if (not finite) or maximum_displacement > material.displacement_criticality_m:
        material.telemetry[3] = material.telemetry[3] + 1.0
        material.telemetry[19] = material.telemetry[18]
        material.telemetry[18] = material.telemetry[17]
        material.telemetry[17] = material.telemetry[16]
        material.telemetry[16] = dt
    material.last_maximum_displacement_m = maximum_displacement
    material.last_maximum_velocity_m_s = maximum_velocity
    return finite, Metrics(
        max_vel=maximum_velocity,
        max_flux=maximum_velocity,
        div_inf=0.0,
        mass_err=0.0,
        error_channels={
            "maximum_substep_displacement_m": maximum_displacement,
        },
    )


def balloon_tire_managed_window(
    material, targets, controller, window_duration, dt_initial
):
    material.telemetry = material.telemetry * 0.0
    material.telemetry[5] = window_duration
    clamp_events_before = controller.clamp_events
    advanced, dt_next, metrics = run_superstep(
        material,
        window_duration,
        dt_initial,
        material.displacement_criticality_m,
        targets,
        controller,
        balloon_tire_managed_advance,
        allow_increase_mid_round=True,
        allow_unresolved=False,
        max_retries=None,
        rollback_threshold_multiplier=2.0,
    )
    completed_window = (
        float(advanced) >= float(window_duration) - 1.0e-15
        and not bool(metrics.hard_failure)
    )
    material.telemetry[0] = float(completed_window)
    material.telemetry[2] = material.telemetry[1] - material.telemetry[3]
    material.telemetry[8] = dt_next
    material.telemetry[9] = advanced
    material.telemetry[10] = window_duration
    material.telemetry[12] = material.displacement_criticality_m
    material.telemetry[13] = (
        material.telemetry[11] / material.displacement_criticality_m
    )
    material.telemetry[15] = controller.clamp_events - clamp_events_before
    return (
        material.state,
        material.output,
        advanced,
        dt_next,
        material.last_maximum_displacement_m,
        material.last_maximum_velocity_m_s,
    )
'''


def vehicle_python_extraction_contract(
    inputs: VehiclePythonCompilationInputs,
):
    """Declare the native tensor boundary without compiling from feeds."""

    from .extraction_contract import ExtractionContract

    values = []
    for parameter, value in inputs.feeds.items():
        if isinstance(value, np.ndarray):
            shape = tuple(map(int, value.shape))
            # A 0-d host array is not a span.  An EMPTY array (some extent
            # zero) still is: its rank and extents are deterministic facts the
            # program keys on (``if edge_nodes.shape[0] == 0``), so declare it
            # exactly rather than dropping it into shapelessness.
            if not shape or any(extent < 0 for extent in shape):
                continue
            values.append({
                "function": inputs.entrypoint,
                "parameter": str(parameter),
                "storage": "span",
                "dtype": str(value.dtype),
                "rank": len(shape),
                "shape": list(shape),
                "python_type": (
                    "src.common.tensors.abstraction.AbstractTensor"
                ),
            })
        elif isinstance(value, (bool, int, float)):
            values.append({
                "function": inputs.entrypoint,
                "parameter": str(parameter),
                "storage": "scalar",
                "dtype": (
                    "bool" if isinstance(value, bool)
                    else "int64" if isinstance(value, int)
                    else "float64"
                ),
                "rank": 0,
                "python_type": f"builtins.{type(value).__name__}",
            })
    policy = ExtractionContract(
        Path(__file__).resolve().parents[2]
        / "extraction_contracts"
        / "program_extraction.yaml"
    )
    contracted = policy.with_program_abi({
        "records": {}, "bindings": [], "values": values,
    })
    return contracted.with_execution_file(
        Path(__file__).resolve().parents[2]
        / "extraction_contracts"
        / "vehicle_full_native_execution.yaml"
    )


def balloon_tire_managed_extraction_contract(
    material: BalloonTireManagedState,
):
    """Declare the real dt-system/tire boundary under the vehicle overlay."""

    from .extraction_contract import ExtractionContract

    policy = ExtractionContract(
        Path(__file__).resolve().parents[2]
        / "extraction_contracts"
        / "program_extraction.yaml"
    )
    base = policy.program_abi.receipt()
    retained_records = {
        name: base["records"][name]
        for name in ("Targets", "STController", "Metrics")
    }
    retained_records["BalloonTireManagedState"] = {
        "identity": (
            "src.compiler.vehicle_python_compilation."
            "BalloonTireManagedState"
        ),
        "fields": {
            name: {
                "storage": "span",
                "dtype": str(value.dtype),
                "rank": int(value.ndim),
                "shape": list(map(int, value.shape)),
                "mutable": name in {"inputs", "state", "output", "telemetry"},
            }
            for name, value in {
                "inputs": material.inputs,
                "state": material.state,
                "output": material.output,
                "wheel_input_indices": material.wheel_input_indices,
                "rest": material.rest,
                "face_vertices": material.face_vertices,
                "face_rest": material.face_rest,
                "face_scatter": material.face_scatter,
                "bending_incidence": material.bending_incidence,
                "bending_scatter": material.bending_scatter,
                "bending_weight": material.bending_weight,
                "vertex_area": material.vertex_area,
                "bead_mask": material.bead_mask,
                "face_material": material.face_material,
                "telemetry": material.telemetry,
            }.items()
        } | {
            "displacement_criticality_m": {
                "storage": "scalar", "dtype": "float64", "mutable": False,
            },
            "last_maximum_displacement_m": {
                "storage": "scalar", "dtype": "float64", "mutable": True,
            },
            "last_maximum_velocity_m_s": {
                "storage": "scalar", "dtype": "float64", "mutable": True,
            },
        },
    }
    retained_bindings = [
        binding for binding in base["bindings"]
        if binding["record"] in {"Targets", "STController", "Metrics"}
    ]
    retained_bindings.append({
        "function": "*", "parameter": "material",
        "record": "BalloonTireManagedState",
    })
    retained_bindings.append({
        "function": "step_with_dt_control_used", "parameter": "state",
        "record": "BalloonTireManagedState",
    })
    abi = {
        "records": retained_records,
        "bindings": retained_bindings,
        "values": [
            {
                "function": "balloon_tire_managed_window",
                "parameter": parameter,
                "storage": "scalar",
                "dtype": "float64",
                "rank": 0,
                "python_type": "builtins.float",
            }
            for parameter in ("window_duration", "dt_initial")
        ],
    }
    return policy.with_program_abi(abi).with_execution_file(
        Path(__file__).resolve().parents[2]
        / "extraction_contracts"
        / "vehicle_full_native_execution.yaml"
    )


@lru_cache(maxsize=32)
def balloon_tire_python_compilation_inputs(
    batch_size: int = BATCH_CAPACITY,
    wheel_names: tuple[str, ...] = WHEEL_NAMES,
    *,
    tire_dimensions: tuple[float, float, float, float, float, float] | None = None,
    pneumatic_mode: str | None = None,
    material_profile: str = "configured",
) -> VehiclePythonCompilationInputs:
    tire = (
        balloon_tire_python_program(
            wheel_names,
            pneumatic_mode=pneumatic_mode,
            material_profile=material_profile,
        )
        if tire_dimensions is None
        else balloon_tire_python_program(
            wheel_names,
            tire_radius_m=tire_dimensions[0],
            tire_section_radius_m=tire_dimensions[1],
            tire_width_m=tire_dimensions[2],
            tire_mass_kg=tire_dimensions[3],
            reference_pressure_pa=tire_dimensions[4],
            rim_radius_m=tire_dimensions[5],
            pneumatic_mode=pneumatic_mode,
            material_profile=material_profile,
        )
    )
    wheel_count = len(wheel_names)
    feeds = {
        "inputs": np.tile(tire.constants["default_input"], (batch_size, 1)),
        "state": np.zeros(
            (batch_size, wheel_count, tire.vertex_count, 6), dtype=np.float64),
        "output": np.zeros((batch_size, wheel_count, 14), dtype=np.float64),
        "wheel_input_indices": tire.constants["wheel_input_indices"],
        "rest": tire.constants["rest"],
        "face_vertices": tire.constants["face_vertices"],
        "face_rest": tire.constants["face_rest"],
        "face_scatter": tire.constants["face_scatter"],
        "bending_incidence": tire.constants["bending_incidence"],
        "bending_scatter": tire.constants["bending_scatter"],
        "bending_weight": tire.constants["bending_weight"],
        "vertex_area": tire.constants["vertex_area"],
        "bead_mask": tire.constants["bead_mask"],
        "face_material": tire.constants["face_material"],
    }
    return VehiclePythonCompilationInputs(
        tire.source, tire.entrypoint, feeds, balloon_tire_linked_process_graphs())


def balloon_tire_default_initialized_state(
    inputs: np.ndarray,
    wheel_input_indices: np.ndarray,
    rest: np.ndarray,
) -> np.ndarray:
    """Materialize the standalone fixture's authored initialization state.

    This is the eager twin of ``balloon_tire_vector_initialize`` in
    :mod:`vehicle_balloon_tire_program`: it prepares the initial material
    receipt only.  Runtime ticks remain the compiled Python program, while the
    full vehicle owns its compiled one-shot initialization gate.
    """

    inputs = np.asarray(inputs, dtype=np.float64)
    wheel_input_indices = np.asarray(wheel_input_indices, dtype=np.int64)
    rest = np.asarray(rest, dtype=np.float64)
    wheel_count = int(wheel_input_indices.shape[0])
    wheel_input = inputs[:, wheel_input_indices.reshape(-1)].reshape(
        (-1, wheel_count, 41)
    )
    basis = wheel_input[:, :, 6:15].reshape((-1, wheel_count, 3, 3))
    hub = wheel_input[:, :, 0:3]
    hub_velocity = wheel_input[:, :, 3:6]
    angle = wheel_input[:, :, 18]
    cosine = np.cos(angle).reshape((-1, wheel_count, 1))
    sine = np.sin(angle).reshape((-1, wheel_count, 1))
    local = rest.reshape((1, 1, -1, 3))
    rotated_local = np.stack((
        cosine * local[:, :, :, 0] - sine * local[:, :, :, 1],
        sine * local[:, :, :, 0] + cosine * local[:, :, :, 1],
        local[:, :, :, 2] * np.ones_like(cosine),
    ), axis=-1)
    radius = np.matmul(rotated_local, basis)
    omega = (
        wheel_input[:, :, 15:18]
        + wheel_input[:, :, 19].reshape((-1, 4, 1)) * basis[:, :, 2, :]
    )
    state = np.zeros(
        (inputs.shape[0], wheel_count, rest.shape[0], 6), dtype=np.float64
    )
    state[:, :, :, 0:3] = hub.reshape((-1, wheel_count, 1, 3)) + radius
    state[:, :, :, 3:6] = (
        hub_velocity.reshape((-1, wheel_count, 1, 3))
        + np.cross(omega.reshape((-1, wheel_count, 1, 3)), radius)
    )
    return state


def balloon_tire_managed_python_compilation_inputs(
    batch_size: int = BATCH_CAPACITY,
    *,
    window_duration: float = 1.0 / 120.0,
    dt_initial: float = 1.0 / 360.0,
    wheel_names: tuple[str, ...] = WHEEL_NAMES,
    tire_dimensions: tuple[float, float, float, float, float, float] | None = None,
    pneumatic_mode: str | None = None,
    material_profile: str = "configured",
) -> VehiclePythonCompilationInputs:
    """Compose the canonical tire with the repository's existing dt system."""

    tire = balloon_tire_python_compilation_inputs(
        batch_size,
        wheel_names,
        tire_dimensions=tire_dimensions,
        pneumatic_mode=pneumatic_mode,
        material_profile=material_profile,
    )
    arrays = {
        name: np.asarray(value).copy()
        for name, value in tire.feeds.items()
    }
    material = BalloonTireManagedState(
        inputs=arrays["inputs"],
        state=balloon_tire_default_initialized_state(
            arrays["inputs"], arrays["wheel_input_indices"], arrays["rest"],
        ),
        output=arrays["output"],
        wheel_input_indices=arrays["wheel_input_indices"],
        rest=arrays["rest"],
        face_vertices=arrays["face_vertices"],
        face_rest=arrays["face_rest"],
        face_scatter=arrays["face_scatter"],
        bending_incidence=arrays["bending_incidence"],
        bending_scatter=arrays["bending_scatter"],
        bending_weight=arrays["bending_weight"],
        vertex_area=arrays["vertex_area"],
        bead_mask=arrays["bead_mask"],
        face_material=arrays["face_material"],
        telemetry=np.zeros((20,), dtype=np.float64),
        displacement_criticality_m=float(arrays["inputs"][0, 21]),
    )
    if material.displacement_criticality_m <= 0.0:
        raise ValueError("balloon tire displacement criticality must be positive")
    return VehiclePythonCompilationInputs(
        source="\n".join((tire.source, BALLOON_TIRE_MANAGED_SOURCE)),
        entrypoint="balloon_tire_managed_window",
        feeds={
            "material": material,
            "targets": Targets(
                cfl=1.0,
                div_max=1.0,
                mass_max=1.0,
                error_limits={
                    "maximum_substep_displacement_m": (
                        material.displacement_criticality_m
                    ),
                },
            ),
            # The managed system may shrink and regrow without an imposed cap.
            "controller": STController(dt_min=None, dt_max=None),
            "window_duration": float(window_duration),
            "dt_initial": float(dt_initial),
        },
        linked_process_graphs=tire.linked_process_graphs,
    )


def lower_balloon_tire_managed_python_ssa(
    *,
    batch_size: int = BATCH_CAPACITY,
    window_duration: float = 1.0 / 120.0,
    dt_initial: float = 1.0 / 360.0,
    wheel_names: tuple[str, ...] = WHEEL_NAMES,
    tire_dimensions: tuple[float, float, float, float, float, float] | None = None,
    pneumatic_mode: str | None = None,
    material_profile: str = "configured",
    progress=None,
) -> VehiclePythonSSALowering:
    """Directly lower tire plus the existing dt system to repository SSA."""

    from .fortran_c_shell import lower_ast_source_to_ssa
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    inputs = balloon_tire_managed_python_compilation_inputs(
        batch_size,
        window_duration=window_duration,
        dt_initial=dt_initial,
        wheel_names=wheel_names,
        tire_dimensions=tire_dimensions,
        pneumatic_mode=pneumatic_mode,
        material_profile=material_profile,
    )
    material = inputs.feeds["material"]
    module, outputs, exports = lower_ast_source_to_ssa(
        inputs.source,
        inputs.entrypoint,
        python_bindings={
            "AbstractTensor": AbstractTensor,
            "Metrics": Metrics,
            "run_superstep": run_superstep,
        },
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        linked_process_graphs=inputs.linked_process_graphs,
        extraction_contract=balloon_tire_managed_extraction_contract(material),
        runtime_closure_only=True,
        name="balloon_tire_managed_python",
        progress=progress,
    )
    matches = tuple(
        name for name in module.functions
        if name.endswith(f"__{inputs.entrypoint}")
    )
    if len(matches) != 1:
        raise RuntimeError(
            "managed balloon lowering did not publish one entrypoint; "
            f"matches={matches!r}"
        )
    return VehiclePythonSSALowering(
        module, matches[0], dict(outputs), tuple(map(str, exports)),
    )


def _managed_native_feeds_by_id(
    lowered: VehiclePythonSSALowering,
    feeds: Mapping[str, Any],
) -> dict[int, Any]:
    """Flatten the authored managed ProgramABI onto physical SSA formals."""

    from .string_table import string_token

    root = lowered.module.functions[lowered.root_name]
    parameter_names = {
        int(value_id): str(name)
        for name, value_id in root.metadata.get("parameter_names", ())
    }
    physical: dict[int, Any] = {}
    for argument in root.args:
        value_id = int(argument.id)
        accounting = dict(argument.accounting or {})
        parameter = accounting.get("program_abi_parameter")
        field = accounting.get("program_abi_field")
        if parameter is None:
            parameter = parameter_names.get(value_id)
        if parameter not in feeds:
            continue
        value = feeds[str(parameter)]
        if field is None:
            physical[value_id] = value
            continue
        path = str(field).split(".")
        held = value
        for member in path:
            if member in {"length", "keys", "values"} and isinstance(
                held, Mapping
            ):
                if member == "length":
                    held = len(held)
                elif member == "keys":
                    held = np.asarray(
                        [string_token(str(key)) for key in held],
                        dtype=np.int64,
                    )
                else:
                    held = np.asarray(tuple(held.values()), dtype=np.float64)
            elif isinstance(held, Mapping):
                held = held[member]
            else:
                held = getattr(held, member)
        # The current scalar ProgramABI has no separate optional-presence limb.
        # Use operation-neutral native representatives for the dt system's
        # numeric optionals: its None floor is 1e-30, while an absent upper
        # bound/limit is positive infinity.  Unlike the former blanket zero,
        # these preserve every guarded min/max and finiteness test and still
        # allow dt_max to become a finite controller-owned value after the first
        # accepted step.
        if held is None and str(argument.dtype or "").casefold() not in {
            "none", "ssa.aggregate",
        }:
            leaf = path[-1] if path else ""
            held = (
                1.0e-30 if leaf == "dt_min"
                else float("inf") if leaf in {"dt_max", "dt_limit"}
                else 0
            )
        physical[value_id] = held
    return physical


def emit_balloon_tire_managed_python_c(
    *,
    batch_size: int = BATCH_CAPACITY,
    window_duration: float = 1.0 / 120.0,
    dt_initial: float = 1.0 / 360.0,
    wheel_names: tuple[str, ...] = WHEEL_NAMES,
    tire_dimensions: tuple[float, float, float, float, float, float] | None = None,
    pneumatic_mode: str | None = None,
    material_profile: str = "configured",
    progress=None,
):
    """Emit the managed tire and repository dt system as native-only C."""

    from .ssa_c_backend import emit_ssa_to_c
    from .work_contract import active_contract

    lowered = lower_balloon_tire_managed_python_ssa(
        batch_size=batch_size,
        window_duration=window_duration,
        dt_initial=dt_initial,
        wheel_names=wheel_names,
        tire_dimensions=tire_dimensions,
        pneumatic_mode=pneumatic_mode,
        material_profile=material_profile,
        progress=progress,
    )
    if active_contract().deployment == "auto":
        # The contract's deployment demand is honored in SSA, where every
        # backend sees it: provable iteration lanes are outlined into
        # callable closures and the C emitter deploys them on turing_pool
        # (serial text remains the in-source fallback). Refusals are honest
        # receipts on the report; they never block emission.
        from .deployment_outlining import outline_independent_iteration_lanes
        from .deployment_compute_selection import select_compute_lanes

        report = outline_independent_iteration_lanes(lowered.module)
        if progress is not None:
            for record in report.outlined:
                progress(f"deployment outline: {record.outline_name}")
            for function, region, reason in report.refused:
                progress(
                    f"deployment refusal: {function} region {region}: {reason}"
                )
        # The compiler, not a product, decides where a compute shader would
        # be valuable: every outlined lane is judged against the GPU
        # dialect and the verdicts ride the module as receipts.
        selection = select_compute_lanes(lowered.module)
        lowered.module.metadata["deployment_compute_selection"] = (
            selection.as_manifest()
        )
        if progress is not None:
            for verdict in selection.verdicts:
                progress(
                    "compute-shader lane "
                    f"{'ELIGIBLE' if verdict.eligible else 'refused'}: "
                    f"{verdict.outline_name}"
                    + (
                        "" if verdict.eligible
                        else " (" + "; ".join(verdict.reasons)[:200] + ")"
                    )
                )
    artifact = emit_ssa_to_c(
        lowered.module,
        lowered.root_name,
        entry_name="balloon_tire_managed_native_c",
    )
    if not artifact.complete:
        raise RuntimeError(
            "managed balloon tire C emission failed: "
            + "; ".join(
                f"{item.operation}: {item.reason}"
                for item in artifact.shortfalls
            )
        )
    return lowered, artifact


def compile_balloon_tire_managed_python_native(
    directory: str | Path,
    *,
    batch_size: int = BATCH_CAPACITY,
    window_duration: float = 1.0 / 120.0,
    dt_initial: float = 1.0 / 360.0,
    optimization: str = "O2",
    progress=None,
):
    """Compile the managed Python source and real dt system to an executable."""

    inputs = balloon_tire_managed_python_compilation_inputs(
        batch_size,
        window_duration=window_duration,
        dt_initial=dt_initial,
    )
    lowered, artifact = emit_balloon_tire_managed_python_c(
        batch_size=batch_size,
        window_duration=window_duration,
        dt_initial=dt_initial,
        progress=progress,
    )
    feeds_by_id = _managed_native_feeds_by_id(lowered, inputs.feeds)
    missing = tuple(
        value_id for value_id in artifact.buffer_order
        if int(value_id) not in feeds_by_id
    )
    if missing:
        raise RuntimeError(
            "managed standalone C material contract has unnamed public "
            f"buffers: {missing!r}"
        )
    executable = artifact.compile_standalone(
        directory, feeds_by_id, optimization=optimization,
    )
    root = lowered.module.functions[lowered.root_name]
    arguments = {int(value.id): value for value in root.args}
    manifest = {
        "schema": "turing.balloon-tire-managed-native.v1",
        "entrypoint": artifact.name,
        "batch_size": int(batch_size),
        "window_duration": float(window_duration),
        "dt_initial": float(dt_initial),
        "optimization": str(optimization),
        "buffers": [],
    }
    for index, (value_id, dtype) in enumerate(zip(
        artifact.buffer_order, artifact.buffer_dtypes,
    )):
        argument = arguments[int(value_id)]
        accounting = dict(argument.accounting or {})
        value = np.asarray(feeds_by_id[int(value_id)])
        parameter = str(accounting.get("program_abi_parameter") or value_id)
        field = accounting.get("program_abi_field")
        manifest["buffers"].append({
            "index": int(index),
            "value_id": int(value_id),
            "name": parameter if field is None else f"{parameter}.{field}",
            "parameter": parameter,
            "field": None if field is None else str(field),
            "dtype": str(dtype),
            "semantic_dtype": str(argument.dtype or ""),
            "shape": list(map(int, value.shape)),
            "element_count": int(value.size),
            "mutable": bool(accounting.get("program_abi_mutable", False)),
            "written": bool(accounting.get("program_abi_field_written", False)),
        })
    (executable.directory / "balloon_tire_managed.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    return executable


def compile_balloon_tire_python_aot(*, batch_size: int = BATCH_CAPACITY,
                                    progress=None):
    """Build the historical captured AOT planning product.

    Native vehicle emission enters through ``lower_balloon_tire_python_ssa``;
    this compatibility adapter remains only for callers explicitly inspecting
    the older dual-IR planning product.
    """

    inputs = balloon_tire_python_compilation_inputs(batch_size)
    return compile_ast_aot(
        inputs.source, inputs.entrypoint, inputs.feeds,
        backend="c", precompile_only=True,
        python_bindings={"AbstractTensor": AbstractTensor},
        linked_process_graphs=inputs.linked_process_graphs,
        progress=progress,
    )


def lower_balloon_tire_python_ssa(*, batch_size: int = BATCH_CAPACITY,
                                  progress=None) -> VehiclePythonSSALowering:
    """Lower the canonical balloon program directly to repository SSA."""

    from .fortran_c_shell import lower_ast_source_to_ssa
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    inputs = balloon_tire_python_compilation_inputs(batch_size)
    contract_inputs = VehiclePythonCompilationInputs(
        inputs.source,
        inputs.entrypoint,
        {
            name: (
                value.astype(np.float64, copy=False)
                if isinstance(value, np.ndarray) and value.dtype != np.float64
                else value
            )
            for name, value in inputs.feeds.items()
        },
        inputs.linked_process_graphs,
    )
    module, outputs, exports = lower_ast_source_to_ssa(
        inputs.source,
        inputs.entrypoint,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        linked_process_graphs=inputs.linked_process_graphs,
        extraction_contract=vehicle_python_extraction_contract(contract_inputs),
        runtime_closure_only=True,
        name="balloon_tire_python",
        progress=progress,
    )
    matches = tuple(
        name for name in module.functions
        if name.endswith(f"__{inputs.entrypoint}")
    )
    if len(matches) != 1:
        raise RuntimeError(
            "canonical balloon lowering did not publish one entrypoint; "
            f"matches={matches!r}"
        )
    return VehiclePythonSSALowering(
        module, matches[0], dict(outputs), tuple(map(str, exports)),
    )


def emit_balloon_tire_python_c(*, batch_size: int = BATCH_CAPACITY,
                               progress=None):
    from .ssa_c_backend import emit_ssa_to_c

    lowered = lower_balloon_tire_python_ssa(
        batch_size=batch_size, progress=progress,
    )
    artifact = emit_ssa_to_c(
        lowered.module, lowered.root_name,
        entry_name="balloon_tire_vector_step",
    )
    if not artifact.complete:
        raise RuntimeError("balloon tire Python C emission failed: " + "; ".join(
            f"{item.operation}: {item.reason}" for item in artifact.shortfalls))
    return artifact


def emit_balloon_tire_python_llvm(*, batch_size: int = BATCH_CAPACITY,
                                  progress=None):
    """Emit LLVM from the same canonical repository SSA used by native C."""

    from .ssa_llvm_backend import emit_ssa_function_to_llvm

    lowered = lower_balloon_tire_python_ssa(
        batch_size=batch_size, progress=progress,
    )
    artifact = emit_ssa_function_to_llvm(
        lowered.module, lowered.root_name,
        entry_name="balloon_tire_vector_step",
    )
    if not artifact.complete:
        raise RuntimeError(
            "balloon tire Python LLVM emission failed: " + "; ".join(
                f"{item.operation}: {item.reason}"
                for item in artifact.shortfalls
            )
        )
    return artifact


def compile_balloon_tire_python_native(
    directory: str | Path,
    *,
    batch_size: int = BATCH_CAPACITY,
    backend: str = "c",
    optimization: str = "O2",
    trace: bool = False,
    progress=None,
):
    """Compile the canonical Python balloon program to a standalone native exe.

    This is the ordinary repository SSA path: the authored Python is lowered
    once and emitted through the requested complete native channel.  The C
    channel is the standalone default; Fortran remains selectable while its
    structural aggregate ABI legalization is completed.  Neither route uses
    the deprecated fused AOT entry or a Python runtime fallback.
    """

    inputs = balloon_tire_python_compilation_inputs(batch_size)
    native_feeds = dict(inputs.feeds)
    native_feeds["state"] = balloon_tire_default_initialized_state(
        native_feeds["inputs"],
        native_feeds["wheel_input_indices"],
        native_feeds["rest"],
    )
    lowered = lower_balloon_tire_python_ssa(
        batch_size=batch_size,
        progress=progress,
    )
    selected = str(backend).casefold()
    if selected == "c":
        from .ssa_c_backend import emit_ssa_to_c

        emitted = emit_ssa_to_c(
            lowered.module,
            lowered.root_name,
            entry_name="balloon_tire_native_c",
        )
        if not emitted.complete:
            raise RuntimeError(
                "balloon tire Python C emission failed: "
                + "; ".join(
                    f"{item.operation}: {item.reason}"
                    for item in emitted.shortfalls
                )
            )
        root = lowered.module.functions[lowered.root_name]
        source_names = {
            int(value_id): str(source_name)
            for source_name, value_id in root.metadata.get(
                "parameter_names", ()
            )
        }
        for value in root.args:
            source_name = (value.accounting or {}).get(
                "program_abi_parameter"
            )
            if source_name:
                source_names.setdefault(int(value.id), str(source_name))
        feeds_by_id = {
            value_id: native_feeds[source_name]
            for value_id, source_name in source_names.items()
            if source_name in native_feeds
        }
        missing = tuple(
            value_id for value_id in emitted.buffer_order
            if int(value_id) not in feeds_by_id
        )
        if missing:
            raise RuntimeError(
                "standalone C material contract has unnamed public buffers: "
                + repr(missing)
            )
        return emitted.compile_standalone(
            directory, feeds_by_id, optimization=optimization,
        )
    if selected != "fortran":
        raise ValueError(
            f"unknown native balloon backend {backend!r}; "
            "expected 'c' or 'fortran'"
        )

    from .fortran_c_shell import compile_fortran_module_c_shell
    from .ssa_fortran_backend import emit_module

    emitted = emit_module(
        lowered.module,
        name="balloon_tire_native",
        outputs=lowered.outputs,
        progress=progress,
    )
    if not emitted.complete:
        raise RuntimeError(
            "balloon tire Python Fortran emission failed: "
            + "; ".join(item.format() for item in emitted.shortfalls)
        )
    return compile_fortran_module_c_shell(
        emitted,
        native_feeds,
        directory,
        entrypoint=lowered.root_name,
        name="balloon_tire_native",
        standalone=True,
        library=False,
        trace=trace,
    )


def vehicle_python_compilation_inputs(
    batch_size: int = BATCH_CAPACITY,
    wheel_names: tuple[str, ...] = WHEEL_NAMES,
    wheel_to_structural_support: tuple[tuple[float, ...], ...] | None = None,
    graph_constants: VehicleGraphConstants | None = None,
    tire_dimensions: tuple[float, float, float, float, float, float] | None = None,
    machine_operator: str = "configured-vehicle",
    machine_input_names: tuple[str, ...] | None = None,
    machine_output_names: tuple[str, ...] | None = None,
    structural_support_positions: tuple[tuple[float, float, float], ...] | None = None,
    tire_pneumatic_mode: str | None = None,
    tire_material_profile: str = "configured",
) -> VehiclePythonCompilationInputs:
    wheel_names = tuple(wheel_names)
    wheel_count = len(wheel_names)
    graph = (vehicle_native_graph_python_program()
             if graph_constants is None
             else VehicleNativeGraphPythonProgram(constants=graph_constants))
    support_count = graph.constants.node_structural_support_binding.shape[1]
    if wheel_to_structural_support is None:
        if wheel_count == support_count:
            support_map = np.eye(wheel_count, support_count, dtype=np.float64)
        elif wheel_count == 0:
            support_map = np.zeros((0, support_count), dtype=np.float64)
        else:
            raise ValueError(
                "non-car wheel axes require an explicit "
                "wheel_to_structural_support mapping")
    else:
        support_map = np.asarray(
            wheel_to_structural_support, dtype=np.float64)
        if support_map.shape != (wheel_count, support_count):
            raise ValueError(
                "wheel_to_structural_support must have shape "
                f"{(wheel_count, support_count)}, got {support_map.shape}")
        if wheel_count and not np.allclose(
                support_map.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "every wheel must conserve load across structural supports")
    tire = (balloon_tire_python_program(
                wheel_names, pneumatic_mode=tire_pneumatic_mode,
                material_profile=tire_material_profile)
            if tire_dimensions is None else balloon_tire_python_program(
                wheel_names,
                tire_radius_m=tire_dimensions[0],
                tire_section_radius_m=tire_dimensions[1],
                tire_width_m=tire_dimensions[2],
                tire_mass_kg=tire_dimensions[3],
                reference_pressure_pa=tire_dimensions[4],
                rim_radius_m=tire_dimensions[5],
                pneumatic_mode=tire_pneumatic_mode,
                material_profile=tire_material_profile))
    if machine_operator not in {"configured-vehicle", "structural-machine"}:
        raise ValueError(f"unknown machine operator {machine_operator!r}")
    vehicle = (compile_symbolic_vehicle_physics()
               if machine_operator == "configured-vehicle" else None)
    if vehicle is not None:
        resolved_input_names = tuple(
            vehicle.function.metadata["argument_names"])
        resolved_output_names = tuple(
            vehicle.function.metadata["output_names"])
        adapter = _adapter_source(vehicle)
    else:
        if machine_input_names is None or machine_output_names is None:
            raise ValueError(
                "structural-machine requires explicit input and output names")
        resolved_input_names = tuple(machine_input_names)
        resolved_output_names = tuple(machine_output_names)
        adapter = _machine_adapter_source(
            resolved_input_names, resolved_output_names)
    material = compile_vehicle_member_material_ssa()
    gas = compile_balloon_gas_ssa()
    membrane = compile_balloon_membrane_face_ssa()
    bead = compile_balloon_bead_implicit_step_ssa()
    source = "\n".join((
        BALLOON_TIRE_VECTOR_SOURCE,
        VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE,
        adapter,
    ))
    vertex_count, edge_count = tire.vertex_count, graph.constants.edge_nodes.shape[0]
    vehicle_input_count = len(resolved_input_names)
    vehicle_output_count = len(resolved_output_names)
    if structural_support_positions is None:
        support_position = np.zeros((support_count, 3), dtype=np.float64)
    else:
        support_position = np.asarray(
            structural_support_positions, dtype=np.float64)
        if support_position.shape != (support_count, 3):
            raise ValueError(
                "structural_support_positions must have shape "
                f"{(support_count, 3)}, got {support_position.shape}")
    feeds = {
        "vehicle_input": np.zeros((batch_size, vehicle_input_count), dtype=np.float64),
        "contact_input": np.zeros((batch_size, wheel_count, 9), dtype=np.float64),
        "fixture_global": np.zeros((batch_size, 10), dtype=np.float64),
        "fixture_wheel": np.zeros((batch_size, wheel_count, 8), dtype=np.float64),
        "fixture_surface": np.zeros((batch_size, 7), dtype=np.float64),
        "tire_input": np.tile(tire.constants["default_input"], (batch_size, 1)),
        "tire_state": np.zeros((batch_size, wheel_count, vertex_count, 6), dtype=np.float64),
        "tire_output": np.zeros((batch_size, wheel_count, 14), dtype=np.float64),
        "tire_previous_hub": np.zeros((batch_size, wheel_count, 3)),
        "tire_previous_basis": np.zeros((batch_size, wheel_count, 3, 3)),
        "tire_previous_angle": np.zeros((batch_size, wheel_count)),
        "tire_previous_plane": np.zeros((batch_size, wheel_count, 2, 3)),
        "rig_points": np.zeros((batch_size, RIG_POINT_COUNT, 21), dtype=np.float64),
        "material_state": np.zeros((batch_size, edge_count, 9), dtype=np.float64),
        "node_reference": graph.constants.node_reference,
        "node_structural_support_binding": (
            graph.constants.node_structural_support_binding),
        "edge_nodes": graph.constants.edge_nodes,
        "edge_geometry": graph.constants.edge_geometry,
        "structural_support_edge_mask": (
            graph.constants.structural_support_edge_mask),
        "wheel_to_structural_support": support_map,
        "structural_support_position": support_position,
        "tire_wheel_input_indices": tire.constants["wheel_input_indices"],
        "tire_rest": tire.constants["rest"],
        "tire_face_vertices": tire.constants["face_vertices"],
        "tire_face_rest": tire.constants["face_rest"],
        "tire_face_scatter": tire.constants["face_scatter"],
        "tire_bending_incidence": tire.constants["bending_incidence"],
        "tire_bending_scatter": tire.constants["bending_scatter"],
        "tire_bending_weight": tire.constants["bending_weight"],
        "tire_vertex_area": tire.constants["vertex_area"],
        "tire_bead_mask": tire.constants["bead_mask"],
        "tire_face_material": tire.constants["face_material"],
        "wheel_assembly_alpha": np.ones((batch_size, wheel_count), dtype=np.float64),
        "tire_assembly_alpha": np.ones((batch_size, wheel_count), dtype=np.float64),
        "compression": np.zeros((batch_size, support_count), dtype=np.float64),
        "compression_velocity": np.zeros((batch_size, support_count), dtype=np.float64),
        "wheel_angle": np.zeros((batch_size, wheel_count), dtype=np.float64),
        "wheel_omega": np.zeros((batch_size, wheel_count), dtype=np.float64),
        "roller_anchor": np.zeros((batch_size, wheel_count, 2), dtype=np.float64),
        "roller_anchor_valid": np.zeros((batch_size, wheel_count, 1), dtype=bool),
        "terrain_triangles": _periodic_terrain_triangles(),
        "pillar_alpha": np.zeros((batch_size, wheel_count), dtype=np.float64),
        "pillar_pose": np.zeros((batch_size, wheel_count, 3), dtype=np.float64),
        "lock_stiffness": np.ones((batch_size,), dtype=np.float64),
        "lock_damping": np.ones((batch_size,), dtype=np.float64),
        "maximum_actuator_force": np.ones((batch_size,), dtype=np.float64),
        "tire_initialized": np.zeros((batch_size,), dtype=bool),
        "tire_history_valid": np.zeros((batch_size,), dtype=bool),
        "outer_dt": np.full((batch_size,), 1.0 / 120.0, dtype=np.float64),
        "microstep_count": 3,
    }
    feeds["fixture_global"][:, :] = np.asarray([
        1.0 / 120.0, 0.0, -9.81, -0.75, 12.0, 1.0, 8.0,
        24_000.0, 1_200.0, 18_000.0,
    ], dtype=np.float64)
    feeds["fixture_surface"][:, 5:7] = 4.0
    # Repository tensor SSA is physically double-backed in the C, LLVM and
    # Fortran lead paths.  Topology indices and masks remain semantic numeric
    # values (gather performs its explicit integer conversion), but presenting
    # an int64/bool host arena as a double kernel pointer would be an ABI lie.
    # Author the entry receipt from the exact physical arrays it will receive.
    feeds = {
        name: (
            value.astype(np.float64, copy=False)
            if isinstance(value, np.ndarray) and value.dtype != np.float64
            else value
        )
        for name, value in feeds.items()
    }
    linked = {
        "vehicle_member_material_step": material.process_graph,
        "balloon_tire_gas": gas.process_graph,
        "balloon_tire_membrane_face": membrane.process_graph,
        "balloon_tire_bead_implicit_step": bead.process_graph,
    }
    if vehicle is not None:
        linked["abstract_ui_vehicle_step"] = vehicle.process_graph
    return VehiclePythonCompilationInputs(
        source, "vehicle_graph_tick_vector", feeds, linked)


def dually_vehicle_python_compilation_inputs(
    batch_size: int = BATCH_CAPACITY,
) -> VehiclePythonCompilationInputs:
    """Specialize the canonical compilable Python validator to the dually.

    This is the native compiler's dually source boundary.  It deliberately
    calls :func:`vehicle_python_compilation_inputs`, the same constructor used
    by the live Python validator, instead of assembling a parallel tire or
    native-shell model.
    """

    from .abstract_ui_vehicles import compile_symbolic_vehicle_physics
    from .vehicle_validator_profiles import dually_validator_profile

    profile = dually_validator_profile()
    vehicle = compile_symbolic_vehicle_physics()
    input_names = tuple(vehicle.function.metadata["argument_names"])
    output_names = tuple(vehicle.function.metadata["output_names"])
    prepared = vehicle_python_compilation_inputs(
        batch_size,
        profile.wheel_names,
        profile.fixture_plan.wheel_to_structural_support,
        profile.graph_constants,
        profile.tire_dimensions,
        "structural-machine",
        input_names,
        output_names,
        profile.structural_support_positions,
        tire_pneumatic_mode=profile.tire_pneumatic_mode,
        tire_material_profile=profile.tire_material_profile,
    )
    # The repository dt system owns subdivision.  One graph invocation is one
    # candidate dt, exactly as in _run_dually_python_profile.
    prepared.feeds["microstep_count"] = 1
    return prepared


def compile_vehicle_python_graph_aot(*, batch_size: int = BATCH_CAPACITY,
                                     progress=None):
    """Enter the existing compiler once; emitters consume its resulting IR."""

    inputs = vehicle_python_compilation_inputs(batch_size)
    return compile_ast_aot(
        inputs.source, inputs.entrypoint, inputs.feeds,
        backend="c", precompile_only=True,
        python_bindings={"AbstractTensor": AbstractTensor},
        linked_process_graphs=inputs.linked_process_graphs,
        progress=progress,
    )


def lower_vehicle_python_graph_ssa(
    *,
    batch_size: int = BATCH_CAPACITY,
    inputs: VehiclePythonCompilationInputs | None = None,
    progress=None,
):
    from .fortran_c_shell import lower_ast_source_to_ssa
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )

    if inputs is None:
        inputs = vehicle_python_compilation_inputs(batch_size)
    extraction_contract = vehicle_python_extraction_contract(inputs)
    compilation_name = "vehicle_python_graph"
    module, outputs, exports = lower_ast_source_to_ssa(
        inputs.source,
        inputs.entrypoint,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name=compilation_name,
        runtime_closure_only=True,
        progress=progress,
        linked_process_graphs=inputs.linked_process_graphs,
        extraction_contract=extraction_contract,
    )
    root_name = f"{compilation_name}__{inputs.entrypoint}"
    if root_name not in module.functions:
        matches = tuple(
            name for name in module.functions
            if name.endswith(f"__{inputs.entrypoint}")
        )
        if len(matches) != 1:
            raise RuntimeError(
                "canonical vehicle lowering did not publish one entrypoint; "
                f"expected {root_name!r}, matches={matches!r}"
            )
        root_name = matches[0]
    return VehiclePythonSSALowering(
        module, root_name, dict(outputs), tuple(map(str, exports)))


def emit_vehicle_python_graph_c_with_lowering(
    *,
    batch_size: int = BATCH_CAPACITY,
    inputs: VehiclePythonCompilationInputs | None = None,
    progress=None,
):
    """Lower once and emit C; return both so a shell can bridge the ABI.

    The native shell needs the root's parameter and named-output ids (to
    bake each ABI buffer from the feeds and to feed state outputs back into
    their input buffers per lane), which only the lowering carries.
    """

    from .ssa_c_backend import emit_ssa_to_c, summarize_c_shortfalls

    lowered = lower_vehicle_python_graph_ssa(
        batch_size=batch_size, inputs=inputs, progress=progress)
    artifact = emit_ssa_to_c(
        lowered.module, lowered.root_name,
        entry_name="vehicle_native_graph_tick")
    if not artifact.complete:
        raise RuntimeError(
            "vehicle Python graph C emission failed: "
            + "; ".join(summarize_c_shortfalls(artifact.shortfalls))
        )
    return lowered, artifact


def emit_vehicle_python_graph_c(
    *,
    batch_size: int = BATCH_CAPACITY,
    inputs: VehiclePythonCompilationInputs | None = None,
    progress=None,
):
    """Emit C only after the complete Python graph has reached repository SSA."""

    _lowered, artifact = emit_vehicle_python_graph_c_with_lowering(
        batch_size=batch_size, inputs=inputs, progress=progress)
    return artifact


def emit_vehicle_python_graph_wasm(*, batch_size: int = BATCH_CAPACITY,
                                   progress=None):
    """Emit Wasm from the identical lowered module used by native C."""

    from .ssa_wasm_backend import emit_ssa_module_to_wasm_core

    lowered = lower_vehicle_python_graph_ssa(
        batch_size=batch_size, progress=progress)
    artifact = emit_ssa_module_to_wasm_core(
        lowered.module, lowered.root_name,
        entry_name="vehicle_native_graph_tick")
    if not artifact.complete:
        raise RuntimeError("vehicle Python graph Wasm emission failed: " + "; ".join(
            item.reason for item in artifact.shortfalls))
    return artifact


def assemble_vehicle_python_module(*, batch_size: int = BATCH_CAPACITY,
                                   progress=None):
    """Emit every vehicle target from one canonical repository-SSA lowering."""

    from .repository_ssa_module import assemble_repository_ssa_module

    lowered = lower_vehicle_python_graph_ssa(
        batch_size=batch_size, progress=progress,
    )
    shader_root = (
        Path(__file__).resolve().parents[3]
        / "spectral-analyzer" / "csrc" / "shaders"
    )
    return assemble_repository_ssa_module(
        lowered.module,
        lowered.root_name,
        entry_name="vehicle_native_graph_tick",
        graphics_shaders=(
            shader_root / "vehicle_scientific.vert.glsl",
            shader_root / "vehicle_scientific.frag.glsl",
        ),
    )


__all__ = ["VehiclePythonCompilationInputs", "VehiclePythonSSALowering",
           "assemble_vehicle_python_module",
           "balloon_tire_python_compilation_inputs",
           "compile_balloon_tire_python_aot",
           "compile_balloon_tire_python_native",
           "compile_vehicle_python_graph_aot",
           "emit_balloon_tire_python_c", "emit_balloon_tire_python_llvm",
           "emit_vehicle_python_graph_c_with_lowering",
           "emit_vehicle_python_graph_c", "emit_vehicle_python_graph_wasm",
           "lower_balloon_tire_python_ssa", "lower_vehicle_python_graph_ssa",
           "vehicle_python_compilation_inputs",
           "vehicle_python_extraction_contract",
           "vehicle_python_runtime_bindings"]
