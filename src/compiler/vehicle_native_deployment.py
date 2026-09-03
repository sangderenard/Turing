"""Native vehicle deployment assembled from the game's canonical kernels.

The native rig is a boundary-condition host, not a second vehicle model.  It
is therefore only allowed to package the complete vehicle transition and wheel
contact functions produced from :mod:`abstract_ui_vehicles`.  In particular,
the reduced suspension-rig kernel is intentionally not accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import sympy

from .abstract_ui_vehicles import (
    compile_torus_plane_contact_arc_c,
    compile_symbolic_vehicle_physics,
    compile_symbolic_vehicle_physics_c,
    compile_wheel_contact_c,
    compile_wheel_contact_ssa,
    load_default_car_configuration,
    _vehicle_mechanical_graph,
)
from .ssa_c_backend import CFunctionArtifact, emit_ssa_function_to_c
from .ir_identities import apply_precision_pipeline
from .symbolic_equation_compiler import (
    SymbolicPublication,
    compile_sympy_equations,
)
from .control_source import ControlProgram, StatementBlock, render_c_shell
from .vehicle_balloon_tire_native import (
    NativeBalloonTireAssembly,
    compile_native_balloon_tire_assembly,
)
from .vehicle_native_assembly import (
    assembly_manifest, combine_c_function_artifacts, compile_brace_on_balance_c,
    compile_leveling_controller_c,
    compile_leveling_sensor_bank_c, compile_wheel_mesh_balance_c,
    native_vehicle_assembly_stages,
)
from .vehicle_mechanical_material import compile_vehicle_member_material_c
from .vehicle_native_graph_program import BATCH_CAPACITY


SCHEMA = "turing.native-vehicle-deployment.v1"
HOLDER_MODES = {"cage-drive": 0, "suspension-test": 1}
FIXTURE_CORNERS = ("front_left", "front_right", "rear_left", "rear_right")


def _render_managed_balloon_tire_validator_source(
    tire: NativeBalloonTireAssembly,
    *,
    batch_size: int = BATCH_CAPACITY,
    window_duration: float,
    dt_initial: float,
    wheel_names: tuple[str, ...] = FIXTURE_CORNERS,
    tire_dimensions: tuple[float, float, float, float, float, float] | None = None,
    pneumatic_mode: str | None = None,
    material_profile: str = "configured",
) -> str:
    """Expose the complete managed AbstractTensor tire through the rig ABI.

    The batch extent is part of the canonical program contract.  Callers may
    select an active extent up to the shared preallocation capacity; this
    adapter must not silently replace that contract with a one-off scalar
    compilation.
    """

    from .vehicle_python_compilation import (
        _managed_native_feeds_by_id,
        balloon_tire_managed_python_compilation_inputs,
        emit_balloon_tire_managed_python_c,
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
    lowered, artifact = emit_balloon_tire_managed_python_c(
        batch_size=batch_size,
        window_duration=window_duration,
        dt_initial=dt_initial,
        wheel_names=wheel_names,
        tire_dimensions=tire_dimensions,
        pneumatic_mode=pneumatic_mode,
        material_profile=material_profile,
    )
    feeds = _managed_native_feeds_by_id(lowered, inputs.feeds)
    root = lowered.module.functions[lowered.root_name]
    arguments = {int(value.id): value for value in root.args}

    from .ssa_storage_requirements import function_storage_requirements

    requirements = function_storage_requirements(
        lowered.module, lowered.root_name
    )
    anonymous_zero_baked: list[int] = []
    field_index: dict[str, int] = {}
    declarations: list[str] = []
    pointers: list[str] = []
    for index, value_id in enumerate(artifact.buffer_order):
        argument = arguments.get(int(value_id))
        accounting = dict(
            (argument.accounting or {}) if argument is not None else {}
        )
        parameter = accounting.get("program_abi_parameter")
        field = accounting.get("program_abi_field")
        if parameter == "material" and field is not None:
            field_index[str(field)] = index
        fed = feeds.get(int(value_id))
        if fed is None:
            # Anonymous captured closure cells (dt-system list arenas and
            # scalars) publish no feed name yet; they start zeroed, sized
            # by the storage-requirements bound. Recorded as receipts --
            # publishing capture names/values is the named lowering task.
            requirement = requirements.get(int(value_id))
            count = (
                int(requirement.element_count)
                if requirement is not None and requirement.element_count
                else 1
            )
            fed = np.zeros(max(1, count), dtype=np.float64)
            anonymous_zero_baked.append(int(value_id))
        # Bake in the buffer's PHYSICAL dtype, never the feed's semantic
        # dtype: a bool mask fed as 128 one-byte flags must become 128
        # float64 cells (the wrapper reads physical storage), not 16 words
        # of packed bytes -- that mis-bake silently masked the bead forces
        # and read past the static (2026-09-01).
        physical = str(artifact.buffer_dtypes[index])
        value = np.ascontiguousarray(
            np.asarray(fed).astype(np.dtype(physical), copy=False)
        )
        raw = value.tobytes(order="C")
        if len(raw) % 8:
            raw += b"\0" * (8 - len(raw) % 8)
        words = np.frombuffer(raw, dtype=np.uint64)
        # C has no zero-length arrays: an EMPTY feed (a legal zero-extent
        # span) still needs one word of backing storage.
        initializer = ",".join(
            f"0x{int(word):016x}ULL" for word in words
        ) or "0ULL"
        declarations.append(
            f"static uint64_t managed_tire_buffer_{index}[{max(1, len(words))}]="
            f"{{{initializer}}};"
        )
        pointers.append(f"managed_tire_buffer_{index}")

    required = {"inputs", "state", "output"}
    missing = required.difference(field_index)
    if missing:
        raise RuntimeError(
            "managed tire validator bridge lacks material buffers: "
            f"{sorted(missing)!r}"
        )
    input_index = field_index["inputs"]
    state_index = field_index["state"]
    output_index = field_index["output"]
    if "telemetry" not in field_index:
        raise RuntimeError(
            "managed tire validator bridge lacks dt diagnostic telemetry"
        )
    telemetry_index = field_index["telemetry"]
    telemetry_buffer_name = f"managed_tire_buffer_{telemetry_index}"
    telemetry_count = int(
        np.asarray(feeds[int(artifact.buffer_order[telemetry_index])]).size
    )
    input_count = len(tire.input_names)
    state_count = int(tire.state_scalar_count)
    output_count = len(tire.output_names)
    if tuple(np.asarray(feeds[int(artifact.buffer_order[input_index])]).shape) != (
        batch_size, input_count,
    ):
        raise RuntimeError("managed tire input shape does not match validator ABI")
    state_shape = tuple(
        np.asarray(feeds[int(artifact.buffer_order[state_index])]).shape
    )
    if (
        state_shape[0] != batch_size
        or math.prod(state_shape[1:]) != state_count
    ):
        raise RuntimeError("managed tire state shape does not match validator ABI")
    output_shape = tuple(
        np.asarray(feeds[int(artifact.buffer_order[output_index])]).shape
    )
    if (
        output_shape[0] != batch_size
        or math.prod(output_shape[1:]) != output_count
    ):
        raise RuntimeError("managed tire output shape does not match validator ABI")

    deployment_receipts = {
        "outlines": [
            record.as_manifest()
            for record in lowered.module.metadata.get(
                "deployment_outlines", {}
            ).values()
        ],
        "compute_selection": lowered.module.metadata.get(
            "deployment_compute_selection"
        ),
        "pooled_regions": [
            list(item) for item in getattr(artifact, "pooled_regions", ())
        ],
        "pool_required": bool(getattr(artifact, "pool_required", False)),
        "anonymous_zero_baked_buffers": anonymous_zero_baked,
        "telemetry_field": "public read-only dt diagnostics",
        "telemetry_count": telemetry_count,
        "managed_tire_batch_capacity": batch_size,
    }

    seat_ti = {name: i for i, name in enumerate(tire.input_names)}
    seat_corners = [
        name.split(".")[0] for name in tire.input_names
        if name.endswith(".hub_position_x")
    ]
    seat_hub_indices = ",".join(
        str(seat_ti[f"{c}.hub_position_x"]) for c in seat_corners)
    seat_velocity_indices = ",".join(
        str(seat_ti[f"{c}.hub_velocity_x"]) for c in seat_corners)
    seat_basis_indices = ",".join(
        str(seat_ti[f"{c}.hub_basis_x_x"]) for c in seat_corners)
    seat_omega_indices = ",".join(
        str(seat_ti[f"{c}.hub_angular_velocity_x"]) for c in seat_corners)
    seat_angle_indices = ",".join(
        str(seat_ti[f"{c}.hub_angle_rad"]) for c in seat_corners)
    seat_spin_indices = ",".join(
        str(seat_ti[f"{c}.hub_angular_velocity_z"]) for c in seat_corners)
    gas_charge_index = seat_ti.get("gas_charge_fraction")
    seat_wheel_stride = state_count // 4
    seat_vertex_count = seat_wheel_stride // 6

    bridge = f'''\n/* Full managed AbstractTensor tire -> canonical validator ABI. */
static void *managed_tire_buffers[{len(pointers)}]={{{",".join(pointers)}}};
static long long managed_tire_extents[1]={{0}};
 TURING_EXPORT void balloon_tire_contact_diagnostics(double *out){{
  memcpy(out,{telemetry_buffer_name},sizeof(double)*{telemetry_count});
}}
TURING_EXPORT void balloon_tire_appendage_defaults(double *out){{
 memcpy(out,managed_tire_buffer_{input_index},sizeof(double)*{input_count});
}}
TURING_EXPORT void balloon_tire_appendage_initialize(const double *in,double *state){{
 /* The reference contract: the pillars hold the wheels FROM THE START.
    The baked ring is the local rest torus at the origin; initialize
    CONSTRUCTS each wheel's ring at its live hub (spin rotation, hub
    basis, hub velocity + omega x r), exactly as the original shell does.
    Nothing creeps into place. */
 static const int seat_hub[4]={{{seat_hub_indices}}};
 static const int seat_vel[4]={{{seat_velocity_indices}}};
 static const int seat_basis[4]={{{seat_basis_indices}}};
 static const int seat_omega[4]={{{seat_omega_indices}}};
 static const int seat_angle[4]={{{seat_angle_indices}}};
 static const int seat_spin[4]={{{seat_spin_indices}}};
 memcpy(managed_tire_buffer_{input_index},in,sizeof(double)*{input_count});
 memcpy(state,managed_tire_buffer_{state_index},sizeof(double)*{state_count});
 for(int w=0;w<4;++w){{
  double ca=cos(in[seat_angle[w]]),sa=sin(in[seat_angle[w]]),spin=in[seat_spin[w]];
  double basis[3][3],omega[3],hub[3],hv[3];
  for(int a=0;a<3;++a){{
   hub[a]=in[seat_hub[w]+a];hv[a]=in[seat_vel[w]+a];
   for(int b=0;b<3;++b)basis[b][a]=in[seat_basis[w]+3*b+a];
   omega[a]=in[seat_omega[w]+a];
  }}
  for(int a=0;a<3;++a)omega[a]+=spin*basis[2][a];
  for(int v=0;v<{seat_vertex_count};++v){{
   double *s=state+{seat_wheel_stride}*w+6*v;
   double lx=ca*s[0]-sa*s[1],ly=sa*s[0]+ca*s[1],lz=s[2],r[3];
   for(int c=0;c<3;++c)r[c]=basis[0][c]*lx+basis[1][c]*ly+basis[2][c]*lz;
   for(int c=0;c<3;++c){{
    s[c]=hub[c]+r[c];
    s[3+c]=hv[c]+omega[(c+1)%3]*r[(c+2)%3]-omega[(c+2)%3]*r[(c+1)%3];
   }}
  }}
 }}
 memcpy(managed_tire_buffer_{state_index},state,sizeof(double)*{state_count});
}}
TURING_EXPORT void balloon_tire_appendage_step(const double *in,double *state,double *out){{
 memcpy(managed_tire_buffer_{input_index},in,sizeof(double)*{input_count});
 memcpy(managed_tire_buffer_{state_index},state,sizeof(double)*{state_count});
 memcpy(managed_tire_buffer_{output_index},out,sizeof(double)*{output_count});
 balloon_tire_managed_native_c(managed_tire_buffers,managed_tire_extents);
 memcpy(state,managed_tire_buffer_{state_index},sizeof(double)*{state_count});
 memcpy(out,managed_tire_buffer_{output_index},sizeof(double)*{output_count});
}}
TURING_EXPORT uintptr_t balloon_tire_web_input_address(void){{
 return (uintptr_t)managed_tire_buffer_{input_index};
}}
TURING_EXPORT uintptr_t balloon_tire_web_state_address(void){{
 return (uintptr_t)managed_tire_buffer_{state_index};
}}
TURING_EXPORT uintptr_t balloon_tire_web_output_address(void){{
 return (uintptr_t)managed_tire_buffer_{output_index};
}}
TURING_EXPORT void balloon_tire_web_step(void){{
 balloon_tire_appendage_step(
  (const double *)managed_tire_buffer_{input_index},
  (double *)managed_tire_buffer_{state_index},
  (double *)managed_tire_buffer_{output_index});
}}
'''
    return "\n".join((
        artifact.source,
        "#include <stdint.h>",
        "#include <string.h>",
        *declarations,
        bridge,
    )), deployment_receipts


def derive_vehicle_rig_rate_hz(config: Any, *, terrain_cells_per_period: int = 8,
                               terrain_period_m: float = 4.0) -> int:
    """Derive physical bandwidth, Nyquist margin, and three solver substeps.

    The three-substep scalar schedule is a host policy, not part of the model
    ABI.  Turing managed scientific time may stand in for this substepper at
    any time, executing the identical graph at its own sampled/superstep time
    with persistent multi-limb arithmetic for peak scientific precision.
    """

    source = config.source
    tire = source["tires"]
    drivetrain = source["drivetrain"]
    top_speed = float(drivetrain["maximum_wheel_speed_rad_s"]) * float(tire["radius"])
    terrain_hz = top_speed / (terrain_period_m / terrain_cells_per_period)
    tire_hz = max(float(tire["longitudinal_deformation_mode_frequency_hz"]),
                  float(tire["lateral_deformation_mode_frequency_hz"]))
    # Cylinder firing is rendered by the PCM observer and is not a forcing
    # term in this averaged BMEP mechanical equation. Scheduling the chassis
    # at its audio bandwidth made the viewer four times slower without adding
    # physical information. Terrain traversal at maximum vehicle speed and
    # the live tire modes are the actual mechanical Nyquist constraints.
    required = 2.0 * max(terrain_hz, tire_hz) * 2.0 * 3.0
    return 1 << max(10, math.ceil(math.log2(required)))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_symbolic_constants_to_double_double(
    constants: Mapping[str, sympy.Basic | float | int],
) -> dict[str, tuple[float, float]]:
    """Represent equation constants in high/low lanes without touching parameters."""

    split = {}
    for name, constant in constants.items():
        precise = sympy.N(sympy.sympify(constant), 80)
        high = float(precise)
        low = float(precise - sympy.Float(high, 80))
        split[str(name)] = (high, low)
    return split


def _canonical_ssa_identity(compilation: Any) -> str:
    """Hash the exact repository-SSA function consumed by every emitter."""

    function = compilation.function
    payload = repr((
        function.name,
        function.args,
        function.blocks,
        function.metadata,
    ))
    return _sha256_text(payload)


def emit_double_double_c(
    compilation: Any, *, entry_name: str, expose_lanes: bool = True,
) -> CFunctionArtifact:
    """Promote one canonical SSA graph to exact two-limb binary64 arithmetic."""

    from src.common.tensors.topological_reducer import PRECISION_SINGULAR_NAMES

    module = copy.deepcopy(compilation.module)
    function = module.functions[compilation.function.name]
    promoted = 0
    for block in function.blocks.values():
        for instruction in block.instrs:
            source_op = str(instruction.op)
            key = "neg" if source_op == "Neg" else source_op
            precision_op = PRECISION_SINGULAR_NAMES.get(key)
            if precision_op is None:
                continue
            instruction.op = precision_op
            instruction.attributes.update({
                "precision_limbs": 2,
                "precision_element": "float64",
                "lowered_from": source_op,
            })
            promoted += 1
    if not promoted:
        raise RuntimeError("double-double promotion found no numerical operations")
    receipt = apply_precision_pipeline(module, two_product_flavor="fma")
    if receipt.get("status") != "lowered":
        raise RuntimeError("double-double precision pipeline did not lower")
    if expose_lanes:
        original_argument_names = tuple(function.metadata.get("argument_names", ()))
        original_arguments = tuple(function.args[:len(original_argument_names)])
        limb_rows = {
            int(value_id): tuple(map(int, limb_ids))
            for value_id, limb_ids in function.metadata.get("precision_lowered_values", ())
        }
        extra_name_by_id = {}
        for name, formal in zip(original_argument_names, original_arguments):
            for limb_index, limb_id in enumerate(limb_rows.get(int(formal.id), ())[1:], 1):
                extra_name_by_id[limb_id] = f"{name}__limb{limb_index}"
        function.metadata["argument_names"] = (
            *original_argument_names,
            *(extra_name_by_id.get(int(formal.id), f"precision_limb_{formal.id}")
              for formal in function.args[len(original_argument_names):]),
        )
        original_output_names = tuple(function.metadata.get("output_names", ()))
        ret = next(
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if str(instruction.op) == "Ret"
        )
        expanded_outputs = []
        expanded_names = []
        values_by_id = {int(value.id): value for value in function.args}
        values_by_id.update({int(instruction.res.id): instruction.res
                             for block in function.blocks.values()
                             for instruction in block.instrs
                             if instruction.res is not None})
        for output_name, output in zip(original_output_names, tuple(ret.args)):
            row = limb_rows.get(int(output.id), (int(output.id),))
            missing = tuple(limb_id for limb_id in row if limb_id not in values_by_id)
            if missing:
                raise RuntimeError(
                    f"double-double output {output_name!r} references unpublished values {missing}")
            expanded_outputs.extend(values_by_id[limb_id] for limb_id in row)
            expanded_names.extend(
                f"{output_name}__{'hi' if index == 0 else f'limb{index}'}"
                for index in range(len(row))
            )
        ret.args = expanded_outputs
        function.metadata["output_names"] = tuple(expanded_names)
        function.metadata["publications"] = ()
    artifact = emit_ssa_function_to_c(
        module, function.name, entry_name=entry_name,
    )
    if not artifact.complete or not artifact.precision_sections:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"double-double C lowering failed: {reasons}")
    return artifact


@dataclass(frozen=True, slots=True)
class WrittenNativeVehicleKernels:
    vehicle_path: Path
    contact_path: Path
    fixture_path: Path
    tire_assembly_path: Path
    tire_scalar_paths: tuple[Path, ...]
    balance_path: Path
    wheel_balance_path: Path
    leveling_controller_path: Path
    material_path: Path
    shell_path: Path
    vertex_shader_path: Path
    fragment_shader_path: Path
    viewer_shell_path: Path
    manifest_path: Path


@lru_cache(maxsize=1)
def compile_vehicle_roller_fixture_ssa():
    """Compile the external roller/carriage dynamics used by the native rig.

    This is laboratory equipment, not an alternate vehicle/contact model.
    Each corner mode is 0 for a passive unilateral hydraulic follower and 1
    for a bidirectional actuator lock.  Per-corner ownership lets one pressure
    sensor hold its calibrated roller while the other wheels continue their
    approach.  The passive force is a compression-only dashpot and is therefore
    incapable of pulling a departing wheel down.
    """

    names = (
        "dt", "mode", "gravity", "floor_y", "carriage_mass",
        "neutral_buoyancy", "passive_damping", "lock_stiffness",
        "lock_damping", "maximum_actuator_force",
        "surface_mode", "terrain_phase_x", "terrain_phase_z",
        "terrain_velocity_x", "terrain_velocity_z",
        "terrain_period_x", "terrain_period_z",
    )
    symbols = {name: sympy.Symbol(name, real=True) for name in names}
    for corner in FIXTURE_CORNERS:
        for stem in (
            "hub_y", "hub_velocity_y", "carriage_y", "carriage_velocity_y",
            "command_y", "command_velocity_y", "roller_reaction", "mode",
        ):
            symbols[f"{stem}_{corner}"] = sympy.Symbol(
                f"{stem}_{corner}", real=True,
            )
    s = symbols
    equations = []
    publications = []
    for corner in FIXTURE_CORNERS:
        mode = sympy.Min(sympy.Max(sympy.Max(s["mode"], s[f"mode_{corner}"]), 0), 1)
        hub_v = s[f"hub_velocity_y_{corner}"]
        carriage_v = s[f"carriage_velocity_y_{corner}"]
        # Only closing motion is damped.  Separation force is exactly zero,
        # which is the mechanical guarantee that the fixture cannot yank the
        # hub downward after a wheel leaves its rollers.
        passive_force = s["passive_damping"] * sympy.Max(0, carriage_v - hub_v)
        lock_raw = (
            s["lock_stiffness"]
            * (s[f"command_y_{corner}"] - s[f"carriage_y_{corner}"])
            + s["lock_damping"]
            * (s[f"command_velocity_y_{corner}"] - carriage_v)
        )
        lock_force = sympy.Min(
            sympy.Max(lock_raw, -s["maximum_actuator_force"]),
            s["maximum_actuator_force"],
        )
        # Passive force is equal-and-opposite across the hub/carriage pair;
        # the lock servo acts on the carriage against its external rail.
        actuator_force = -(1 - mode) * passive_force + mode * lock_force
        gravity_force = s["carriage_mass"] * s["gravity"]
        compensation_force = -s["neutral_buoyancy"] * gravity_force
        acceleration = (
            gravity_force + compensation_force + actuator_force
            - s[f"roller_reaction_{corner}"]
        ) / s["carriage_mass"]
        candidate_velocity = carriage_v + s["dt"] * acceleration
        candidate_y = s[f"carriage_y_{corner}"] + s["dt"] * candidate_velocity
        next_y = sympy.Max(s["floor_y"], candidate_y)
        # Derive velocity from the accepted displacement so no hidden residual
        # accumulates behind the floor constraint.
        next_velocity = (next_y - s[f"carriage_y_{corner}"]) / s["dt"]
        values = {
            f"carriage_y_{corner}_next": next_y,
            f"carriage_velocity_y_{corner}_next": next_velocity,
            f"fixture_actuator_force_{corner}": actuator_force,
            f"fixture_hub_force_{corner}": (1 - mode) * passive_force,
            f"fixture_compensation_force_{corner}": compensation_force,
        }
        for output_name, expression in values.items():
            equations.append(sympy.Eq(sympy.Symbol(output_name, real=True), expression, evaluate=False))
            publications.append(SymbolicPublication(output_name, f"rig.fixture.{output_name}"))
    patch_values = {
        "terrain_phase_x_next": s["terrain_phase_x"] + s["dt"] * s["terrain_velocity_x"],
        "terrain_phase_z_next": s["terrain_phase_z"] + s["dt"] * s["terrain_velocity_z"],
        "surface_mode_state": sympy.Min(sympy.Max(s["surface_mode"], 0), 1),
        "terrain_period_x_state": sympy.Max(s["terrain_period_x"], sympy.Float("0.01")),
        "terrain_period_z_state": sympy.Max(s["terrain_period_z"], sympy.Float("0.01")),
    }
    for output_name, expression in patch_values.items():
        equations.append(sympy.Eq(
            sympy.Symbol(output_name, real=True), expression, evaluate=False,
        ))
        publications.append(SymbolicPublication(
            output_name, f"rig.fixture.surface.{output_name}",
        ))
    compiled = compile_sympy_equations(
        tuple(equations), name="vehicle_roller_fixture_step",
        publications=tuple(publications), dtype="float64",
    )
    return compiled


def compile_vehicle_roller_fixture_c(*, double_double: bool = False) -> CFunctionArtifact:
    """Legacy comparison emitter for the symbolic fixture graph."""

    compiled = compile_vehicle_roller_fixture_ssa()
    artifact = (
        emit_double_double_c(
            compiled, entry_name="vehicle_roller_fixture_step_dd",
        )
        if double_double
        else emit_ssa_function_to_c(
            compiled.module, compiled.function.name,
            entry_name="vehicle_roller_fixture_step",
        )
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"roller fixture does not lower to C: {reasons}")
    return artifact


def _mechanical_edge_endpoints(edge: Mapping[str, Any]) -> tuple[str, str]:
    """Return the two structural nodes without imposing one graph spelling.

    The legacy car graph names the endpoints ``a`` and ``b``.  General
    machine graphs use the schema-neutral ``nodes`` pair.  Native deployment
    consumes the mechanical relationship, not either author's serialization
    convention.
    """

    if "a" in edge and "b" in edge:
        return str(edge["a"]), str(edge["b"])
    endpoints = edge.get("nodes")
    if isinstance(endpoints, (list, tuple)) and len(endpoints) == 2:
        return str(endpoints[0]), str(endpoints[1])
    identity = str(edge.get("identity", "<unnamed mechanical edge>"))
    raise ValueError(f"mechanical edge {identity!r} does not have two endpoints")


def _render_native_material_bank(
    vehicle: CFunctionArtifact,
    material: CFunctionArtifact,
    *,
    mechanical_graph: Mapping[str, Any] | None = None,
) -> tuple[str, Mapping[str, Any]]:
    """Emit persistent compiled material state for every damage-bearing edge."""

    graph = (
        _vehicle_mechanical_graph(load_default_car_configuration())
        if mechanical_graph is None
        else mechanical_graph
    )
    nodes = tuple(graph["nodes"])
    node_index = {node["identity"]: index for index, node in enumerate(nodes)}
    edges = tuple(edge for edge in graph["edges"] if edge.get("damage", {}).get(
        "model") == "elastic-plastic-member-with-shear-fracture")
    vi = {name: index for index, name in enumerate(vehicle.input_names)}
    mi = {name: index for index, name in enumerate(material.input_names)}
    mo = {name: index for index, name in enumerate(material.output_names)}
    required = {
        "dt", "position_x", "position_y", "position_z", "roll", "pitch", "yaw",
        "velocity_x", "velocity_y", "velocity_z", "roll_velocity", "pitch_velocity",
        "yaw_velocity", *(f"compression_{corner}" for corner in FIXTURE_CORNERS),
        *(f"compression_velocity_{corner}" for corner in FIXTURE_CORNERS),
        *(f"material_plastic_set_{corner}" for corner in FIXTURE_CORNERS),
        *(f"material_survival_{corner}" for corner in FIXTURE_CORNERS),
    }
    if not required <= vi.keys():
        raise RuntimeError(f"native material graph ABI missing {sorted(required - vi.keys())}")

    def c_rows(rows: list[list[float | int]]) -> str:
        return "{" + ",".join("{" + ",".join(
            str(value) if isinstance(value, int) else f"{float(value):.17g}"
            for value in row) + "}" for row in rows) + "}"

    node_rows: list[list[float | int]] = []
    for node in nodes:
        coordinate = str(node.get("generalized_coordinate") or "")
        corner = next((index for index, name in enumerate(FIXTURE_CORNERS)
                       if coordinate == f"compression_{name}"), -1)
        moves_with = node.get("moves_with")
        chassis_bound = (node.get("fixed_to") == "chassis" or moves_with == "chassis")
        binding = corner + 1 if corner >= 0 and not chassis_bound else 0
        node_rows.append([*(float(value) for value in node["reference_position"]), binding])

    edge_rows: list[list[float | int]] = []
    edge_corners: list[int] = []
    for edge in edges:
        damage = edge["damage"]
        endpoint_a, endpoint_b = _mechanical_edge_endpoints(edge)
        authored_rest = float(damage.get("natural_rest_length", edge["rest_length"]))
        reference_rest = math.dist(
            nodes[node_index[endpoint_a]]["reference_position"],
            nodes[node_index[endpoint_b]]["reference_position"],
        )
        # Coincident constraints are joints, bearings, and bushing cartridges,
        # not zero-length axial beams. Their translational/angular junction
        # coordinates need a junction damage operator; feeding |b-a|/1e-9 to
        # the beam return map fractures them at the authored rest pose.
        a = node_index[endpoint_a]
        b = node_index[endpoint_b]
        binding_pair = {int(node_rows[a][3]), int(node_rows[b][3])}
        # The reduced vehicle coordinate translates an unsprung corner; it
        # does not solve the lateral arc that keeps an A-arm or halfshaft at
        # constant length.  Cross-binding spans therefore cannot use that
        # display/reduced coordinate as axial beam strain.
        axial_kinematic = (1.0 if authored_rest > 1.0e-8
                           and reference_rest > 1.0e-8
                           and len(binding_pair) == 1 else 0.0)
        rest = authored_rest if axial_kinematic else 1.0
        radius = max(1.0e-4, float(edge.get("radius", 0.018)))
        yield_stress = float(damage.get("yield_strength_pa", 350_000_000.0))
        authored_yield_force = max(1.0, float(damage.get("axial_yield_force_n", 1.0)))
        area = float(damage.get("section_area_m2", authored_yield_force / yield_stress))
        area = max(area, math.pi * radius * radius * 0.04)
        youngs = float(damage.get("youngs_modulus_pa", 205_000_000_000.0))
        shear = youngs / 2.6
        fracture_strain = max(float(damage.get("fracture_strain", 0.075)), 1.0e-4)
        bindings = {int(node_rows[a][3]), int(node_rows[b][3])} - {0}
        corner = next(iter(bindings)) - 1 if len(bindings) == 1 else -1
        edge_corners.append(corner)
        edge_rows.append([
            a, b, corner, rest, area, area * rest, youngs, shear, yield_stress,
            max(yield_stress * 1.35, yield_stress + 1.0), youngs * 0.01,
            fracture_strain, 0.35, youngs * 5.0e-5, axial_kinematic,
        ])

    corner_edge_counts = [sum(corner == index for corner in edge_corners)
                          for index in range(len(FIXTURE_CORNERS))]
    source = f'''
#define VEHICLE_NATIVE_MATERIAL_NODE_COUNT {len(nodes)}
#define VEHICLE_NATIVE_MATERIAL_EDGE_COUNT {len(edges)}
#define VEHICLE_NATIVE_MATERIAL_STATE_STRIDE 9
#define VEHICLE_NATIVE_MATERIAL_DIAGNOSTIC_STRIDE 8
typedef struct {{double reference[3];int binding;}} VehicleNativeMaterialNode;
typedef struct {{int a,b,corner;double rest,area,volume,youngs,shear,yield_stress,ultimate,hardening,fracture_plastic,fragility,viscosity,axial_kinematic;}} VehicleNativeMaterialEdge;
static const VehicleNativeMaterialNode vehicle_native_material_nodes[VEHICLE_NATIVE_MATERIAL_NODE_COUNT]={c_rows(node_rows)};
static const VehicleNativeMaterialEdge vehicle_native_material_edges[VEHICLE_NATIVE_MATERIAL_EDGE_COUNT]={c_rows(edge_rows)};
static double vehicle_native_material_state[VEHICLE_NATIVE_MATERIAL_EDGE_COUNT][VEHICLE_NATIVE_MATERIAL_STATE_STRIDE];
static double vehicle_native_material_diagnostic[VEHICLE_NATIVE_MATERIAL_EDGE_COUNT][VEHICLE_NATIVE_MATERIAL_DIAGNOSTIC_STRIDE];
void {material.name}(const double *,double *);
static void vehicle_native_material_node_state(const double *in,int index,double *p,double *v){{
 const VehicleNativeMaterialNode *n=&vehicle_native_material_nodes[index];
 const double cr=cos(in[{vi['roll']}]),sr=sin(in[{vi['roll']}]),cp=cos(in[{vi['pitch']}]),sp=sin(in[{vi['pitch']}]),cy=cos(in[{vi['yaw']}]),sy=sin(in[{vi['yaw']}]);
 const double x1=cy*n->reference[0]-sy*n->reference[2],z1=sy*n->reference[0]+cy*n->reference[2],y1=cr*n->reference[1]-sr*z1;
 double r[3]={{cp*x1-sp*y1,sp*x1+cp*y1,sr*n->reference[1]+cr*z1}};
 const double omega[3]={{in[{vi['roll_velocity']}],-in[{vi['yaw_velocity']}],in[{vi['pitch_velocity']}]}},body_v[3]={{in[{vi['velocity_x']}],in[{vi['velocity_y']}],in[{vi['velocity_z']}]}};
 double q=0.0,qd=0.0;if(n->binding>0){{const int c=n->binding-1;const int qi[4]={{{','.join(str(vi[f'compression_{c}']) for c in FIXTURE_CORNERS)}}},qdi[4]={{{','.join(str(vi[f'compression_velocity_{c}']) for c in FIXTURE_CORNERS)}}};q=in[qi[c]];qd=in[qdi[c]];for(int a=0;a<3;++a)r[a]+=q*(a==0?-sp*cr:(a==1?cp*cr:sr));}}
 p[0]=in[{vi['position_x']}]+r[0];p[1]=in[{vi['position_y']}]+r[1];p[2]=in[{vi['position_z']}]+r[2];
 v[0]=body_v[0]+omega[1]*r[2]-omega[2]*r[1]-qd*sp*cr;v[1]=body_v[1]+omega[2]*r[0]-omega[0]*r[2]+qd*cp*cr;v[2]=body_v[2]+omega[0]*r[1]-omega[1]*r[0]+qd*sr;
}}
static void vehicle_native_material_step(double *in){{
 double corner_plastic[4]={{0,0,0,0}},corner_survival[4]={{1,1,1,1}};
 for(int e=0;e<VEHICLE_NATIVE_MATERIAL_EDGE_COUNT;++e){{const VehicleNativeMaterialEdge *g=&vehicle_native_material_edges[e];double pa[3],pb[3],va[3],vb[3];vehicle_native_material_node_state(in,g->a,pa,va);vehicle_native_material_node_state(in,g->b,pb,vb);double d[3],dv[3],l2=0,dotv=0;for(int a=0;a<3;++a){{d[a]=pb[a]-pa[a];dv[a]=vb[a]-va[a];l2+=d[a]*d[a];dotv+=d[a]*dv[a];}}double length=sqrt(fmax(l2,1e-30)),axial=g->axial_kinematic*(length-g->rest)/g->rest,rate=g->axial_kinematic*dotv/(length*g->rest),mi_[{len(material.input_names)}]={{0}},mo_[{len(material.output_names)}]={{0}},*s=vehicle_native_material_state[e];
 mi_[{mi['dt']}]=in[{vi['dt']}];mi_[{mi['axial_strain']}]=axial;mi_[{mi['bending_strain']}]=0;mi_[{mi['shear_strain']}]=0;mi_[{mi['axial_strain_rate']}]=rate;mi_[{mi['bending_strain_rate']}]=0;mi_[{mi['shear_strain_rate']}]=0;mi_[{mi['plastic_axial_previous']}]=s[0];mi_[{mi['plastic_bending_previous']}]=s[1];mi_[{mi['plastic_shear_previous']}]=s[2];mi_[{mi['accumulated_plastic_strain_previous']}]=s[3];mi_[{mi['dissipated_energy_previous']}]=s[4];mi_[{mi['failed_previous']}]=s[5];mi_[{mi['youngs_modulus_pa']}]=g->youngs;mi_[{mi['shear_modulus_pa']}]=g->shear;mi_[{mi['initial_yield_stress_pa']}]=g->yield_stress;mi_[{mi['ultimate_stress_pa']}]=g->ultimate;mi_[{mi['hardening_modulus_pa']}]=g->hardening;mi_[{mi['fracture_plastic_strain']}]=g->fracture_plastic;mi_[{mi['hardening_fragility']}]=g->fragility;mi_[{mi['material_volume_m3']}]=g->volume;mi_[{mi['axial_viscosity_pa_s']}]=g->viscosity;mi_[{mi['bending_viscosity_pa_s']}]=g->viscosity;mi_[{mi['shear_viscosity_pa_s']}]=g->viscosity;{material.name}(mi_,mo_);
 s[0]=mo_[{mo['plastic_axial_next']}];s[1]=mo_[{mo['plastic_bending_next']}];s[2]=mo_[{mo['plastic_shear_next']}];s[3]=mo_[{mo['accumulated_plastic_strain_next']}];s[4]=mo_[{mo['dissipated_energy_next']}];s[5]=mo_[{mo['failed_next']}];s[6]=axial;s[7]=0;s[8]=0;double *o=vehicle_native_material_diagnostic[e];o[0]=axial;o[1]=s[0];o[2]=s[3];o[3]=mo_[{mo['work_hardening_next']}];o[4]=s[5];o[5]=mo_[{mo['axial_stress_pa']}];o[6]=mo_[{mo['fracture_demand']}];o[7]=s[4];if(g->corner>=0){{corner_plastic[g->corner]=fmax(corner_plastic[g->corner],fabs(s[0])*g->rest);corner_survival[g->corner]=fmin(corner_survival[g->corner],1.0-s[5]);}}
 }}
 const int pi[4]={{{','.join(str(vi[f'material_plastic_set_{c}']) for c in FIXTURE_CORNERS)}}},si[4]={{{','.join(str(vi[f'material_survival_{c}']) for c in FIXTURE_CORNERS)}}};for(int c=0;c<4;++c){{in[pi[c]]=corner_plastic[c];in[si[c]]=corner_survival[c];}}
}}
TURING_EXPORT int vehicle_native_material_edge_count(void){{return VEHICLE_NATIVE_MATERIAL_EDGE_COUNT;}}
TURING_EXPORT void vehicle_native_material_diagnostics(double *out){{memcpy(out,vehicle_native_material_diagnostic,sizeof(vehicle_native_material_diagnostic));}}
TURING_EXPORT void vehicle_native_material_state_get(double *out){{memcpy(out,vehicle_native_material_state,sizeof(vehicle_native_material_state));}}
TURING_EXPORT void vehicle_native_material_state_set(const double *in){{memcpy(vehicle_native_material_state,in,sizeof(vehicle_native_material_state));}}
'''
    return source, {
        "edge_count": len(edges), "node_count": len(nodes),
        "state_scalar_count": len(edges) * 9,
        "diagnostic_scalar_count": len(edges) * 8,
        "corner_edge_counts": dict(zip(FIXTURE_CORNERS, corner_edge_counts)),
    }


def render_native_vehicle_tick_shell(
    vehicle: CFunctionArtifact,
    contact: CFunctionArtifact,
    fixture: CFunctionArtifact,
    tire: NativeBalloonTireAssembly,
    material: CFunctionArtifact | None = None,
    *,
    tire_microsteps: int = 48,
    mechanical_graph: Mapping[str, Any] | None = None,
) -> str:
    """Render one closed graph tick using the exact balloon-skin appendage.

    ``contact`` supplies only the established wheel attachment/heading ABI to
    the host.  No force law from the legacy contact kernel is called here.
    The persistent tire state and its stable microstep schedule are owned by
    this generated C shell, immediately adjacent to the canonical vehicle
    transition, so the native rig cannot silently substitute a second model.
    """

    if material is None:
        material = compile_vehicle_member_material_c()
    vi = {name: index for index, name in enumerate(vehicle.input_names)}
    ci = {name: index for index, name in enumerate(contact.input_names)}
    fi = {name: index for index, name in enumerate(fixture.input_names)}
    fo = {name: index for index, name in enumerate(fixture.output_names)}
    ti = {name: index for index, name in enumerate(tire.input_names)}
    gas_charge_index = ti.get("gas_charge_fraction")
    to = {name: index for index, name in enumerate(tire.output_names)}
    tire_output_stride = len(tire.output_names) // len(FIXTURE_CORNERS)
    bending_energy_offset = to[f"{FIXTURE_CORNERS[0]}.bending_energy_j"]
    required_vehicle = {
        *(f"contact_normal_force_{corner}" for corner in FIXTURE_CORNERS),
        *(f"tire_reaction_torque_{corner}" for corner in FIXTURE_CORNERS),
        *(f"compression_{corner}" for corner in FIXTURE_CORNERS),
        *(f"compression_velocity_{corner}" for corner in FIXTURE_CORNERS),
        *(f"wheel_omega_{corner}" for corner in FIXTURE_CORNERS),
        *(f"wheel_angle_{corner}" for corner in FIXTURE_CORNERS),
        "position_x", "position_y", "position_z", "velocity_y",
        "velocity_x", "velocity_z", "roll", "pitch", "yaw",
        "roll_velocity", "pitch_velocity", "yaw_velocity",
        "assembly_alpha_drivetrain",
        *(f"assembly_alpha_{corner}" for corner in FIXTURE_CORNERS),
        "contact_wrench_force_x", "contact_wrench_force_y", "contact_wrench_force_z",
        "contact_wrench_torque_x", "contact_wrench_torque_y", "contact_wrench_torque_z",
    }
    required_contact = {
        *(f"forward_{axis}" for axis in "xyz"),
        *(f"attachment_{axis}" for axis in "xyz"),
        "support", "tire_radial_compression", "tire_radial_velocity",
    }
    required_tire = {
        "dt", "gravity_y",
        *(f"{corner}.hub_position_{axis}" for corner in FIXTURE_CORNERS for axis in "xyz"),
        *(f"{corner}.hub_velocity_{axis}" for corner in FIXTURE_CORNERS for axis in "xyz"),
        *(f"{corner}.hub_basis_{local}_{world}" for corner in FIXTURE_CORNERS
          for local in "xyz" for world in "xyz"),
        *(f"{corner}.hub_angular_velocity_{axis}" for corner in FIXTURE_CORNERS for axis in "xyz"),
        *(f"{corner}.hub_angle_rad" for corner in FIXTURE_CORNERS),
        *(f"{corner}.hub_angular_velocity_z" for corner in FIXTURE_CORNERS),
        *(f"{corner}.surface_kind" for corner in FIXTURE_CORNERS),
        *(f"{corner}.cylinder_radius_m" for corner in FIXTURE_CORNERS),
        *(f"{corner}.plane_count" for corner in FIXTURE_CORNERS),
        *(f"{corner}.plane_{plane}_{quantity}_{axis}"
          for corner in FIXTURE_CORNERS for plane in range(2)
          for quantity in ("point", "normal", "velocity") for axis in "xyz"),
    }
    if (not required_vehicle <= vi.keys() or not required_contact <= ci.keys()
            or not required_tire <= ti.keys()):
        raise RuntimeError(
            "canonical vehicle/balloon ABI cannot form native graph tick; missing "
            f"vehicle={sorted(required_vehicle - vi.keys())}, "
            f"contact={sorted(required_contact - ci.keys())}, "
            f"tire={sorted(required_tire - ti.keys())}"
        )
    # The balloon membrane's measured stable bandwidth is audio-rate.  At the
    # former 16,384 Hz schedule a symmetric 0.1% volume perturbation collapsed
    # to roughly half volume and generated a non-physical rim moment; the same
    # test at 49,152 Hz restored to reference volume.  Keep the rate explicit
    # in the emitted graph boundary so no host wall-clock cadence can lower it.
    if tire_microsteps < 1:
        raise ValueError("tire_microsteps must be positive")
    roller_radius = 0.13
    roller_offset = 0.18
    plane_point_indices = "{{" + "},{".join(
        ",".join(str(ti[f"{corner}.plane_{plane}_point_x"])
                 for plane in range(2))
        for corner in FIXTURE_CORNERS
    ) + "}}"
    hub_position_indices = "{{" + "},{".join(
        ",".join(str(ti[f"{corner}.hub_position_{axis}"]) for axis in "xyz")
        for corner in FIXTURE_CORNERS
    ) + "}}"
    hub_basis_indices = "{{" + "},{".join(
        "{" + "},{".join(
            ",".join(str(ti[f"{corner}.hub_basis_{local}_{world}"]) for world in "xyz")
            for local in "xyz"
        ) + "}"
        for corner in FIXTURE_CORNERS
    ) + "}}"
    hub_angle_indices = "{" + ",".join(
        str(ti[f"{corner}.hub_angle_rad"]) for corner in FIXTURE_CORNERS
    ) + "}"
    material_source, _material_metadata = _render_native_material_bank(
        vehicle, material, mechanical_graph=mechanical_graph)
    lines = [
        f"double fixture_out[{len(fixture.output_names)}];",
        "double force[3] = {0.0, 0.0, 0.0};",
        "double torque[3] = {0.0, 0.0, 0.0};",
        "double pillar_force_y[4] = {0.0, 0.0, 0.0, 0.0};",
        "vehicle_native_material_step(vehicle_in);",
        "vehicle_native_apply_rig_points(vehicle_in,force,torque);",
        "const int compression_index[4] = {" + ",".join(
            str(vi[f"compression_{corner}"]) for corner in FIXTURE_CORNERS
        ) + "};",
        "const int compression_velocity_index[4] = {" + ",".join(
            str(vi[f"compression_velocity_{corner}"]) for corner in FIXTURE_CORNERS
        ) + "};",
        "const int fixture_hub_velocity_index[4] = {" + ",".join(
            str(fi[f"hub_velocity_y_{corner}"]) for corner in FIXTURE_CORNERS
        ) + "};",
        "const int fixture_carriage_index[4] = {" + ",".join(
            str(fi[f"carriage_y_{corner}"]) for corner in FIXTURE_CORNERS
        ) + "};",
        "const int fixture_carriage_velocity_index[4] = {" + ",".join(
            str(fi[f"carriage_velocity_y_{corner}"]) for corner in FIXTURE_CORNERS
        ) + "};",
        "if(!tire_defaults_loaded){balloon_tire_appendage_defaults(tire_in);tire_defaults_loaded=1;}",
        *([f"tire_in[{gas_charge_index}]=vehicle_native_tire_gas_charge;"]
          if gas_charge_index is not None else []),
        # Tire weight already belongs to the canonical unsprung generalized
        # coordinate. Vertex mass remains live for deformation/contact, while
        # zero gravity here prevents counting those same 14 kg twice.
        f"tire_in[{ti['gravity_y']}]=0.0;",
        *([f"tire_in[{ti['reference_pressure_pa']}]=contact_in[{ci['tire_pressure']}];"]
          if "reference_pressure_pa" in ti and "tire_pressure" in ci else []),
        "for (int corner = 0; corner < 4; ++corner) {",
        f"    double *cin = contact_in + corner * {len(contact.input_names)};",
        f"    const double derived_hub_x = vehicle_in[{vi['position_x']}] + cin[{ci['attachment_x']}];",
        f"    const double derived_hub_y = vehicle_in[{vi['position_y']}] + cin[{ci['attachment_y']}] + vehicle_in[compression_index[corner]];",
        f"    const double derived_hub_z = vehicle_in[{vi['position_z']}] + cin[{ci['attachment_z']}];",
        "    const double pillar_alpha=vehicle_native_pillar_alpha[corner];",
        f"    const double cr=cos(vehicle_in[{vi['roll']}]),sr=sin(vehicle_in[{vi['roll']}]),cp=cos(vehicle_in[{vi['pitch']}]),sp=sin(vehicle_in[{vi['pitch']}]),cy=cos(vehicle_in[{vi['yaw']}]),sy=sin(vehicle_in[{vi['yaw']}]);",
        "    const double body_basis[3][3]={{cp*cy+sp*sr*sy,sp*cy-cp*sr*sy,cr*sy},{-sp*cr,cp*cr,sr},{-cp*sy+sp*sr*cy,-sp*sy-cp*sr*cy,cr*cy}};",
        f"    const double local_attachment[3]={{cin[{ci['attachment_x']}],cin[{ci['attachment_y']}],cin[{ci['attachment_z']}]}};double rotated_attachment[3]={{0,0,0}};for(int local=0;local<3;++local)for(int world=0;world<3;++world)rotated_attachment[world]+=body_basis[local][world]*local_attachment[local];",
        f"    const double articulated_hub[3]={{vehicle_in[{vi['position_x']}]+rotated_attachment[0],vehicle_in[{vi['position_y']}]+rotated_attachment[1]+vehicle_in[compression_index[corner]],vehicle_in[{vi['position_z']}]+rotated_attachment[2]}};",
        "    const double hub_x=articulated_hub[0],hub_y=articulated_hub[1],hub_z=articulated_hub[2];",
        "    if(!vehicle_native_roller_anchor_valid[corner]){vehicle_native_roller_anchor[corner][0]=hub_x;vehicle_native_roller_anchor[corner][1]=hub_z;vehicle_native_roller_anchor_valid[corner]=1;}",
        f"    const double body_omega[3]={{vehicle_in[{vi['roll_velocity']}],-vehicle_in[{vi['yaw_velocity']}],vehicle_in[{vi['pitch_velocity']}]}};",
        f"    const double body_velocity[3]={{vehicle_in[{vi['velocity_x']}],vehicle_in[{vi['velocity_y']}],vehicle_in[{vi['velocity_z']}]}};",
        "    double hub_velocity[3]={body_velocity[0]+body_omega[1]*rotated_attachment[2]-body_omega[2]*rotated_attachment[1],body_velocity[1]+body_omega[2]*rotated_attachment[0]-body_omega[0]*rotated_attachment[2],body_velocity[2]+body_omega[0]*rotated_attachment[1]-body_omega[1]*rotated_attachment[0]};for(int world=0;world<3;++world)hub_velocity[world]+=vehicle_in[compression_velocity_index[corner]]*body_basis[1][world];",
        f"    pillar_force_y[corner]=pillar_alpha*(fixture_in[{fi['lock_stiffness']}]*(vehicle_native_pillar_pose[corner][1]-articulated_hub[1])-fixture_in[{fi['lock_damping']}]*hub_velocity[1]);double pillar_limit=fixture_in[{fi['maximum_actuator_force']}];if(pillar_force_y[corner]>pillar_limit)pillar_force_y[corner]=pillar_limit;if(pillar_force_y[corner]<-pillar_limit)pillar_force_y[corner]=-pillar_limit;vehicle_native_pillar_reaction_y[corner]=-pillar_force_y[corner];",
        "    fixture_in[fixture_hub_velocity_index[corner]] = hub_velocity[1];",
        "    const int hub_position_index[4][3]={{" + "},{".join(
            ",".join(str(ti[f"{corner}.hub_position_{axis}"]) for axis in "xyz")
            for corner in FIXTURE_CORNERS) + "}};",
        "    const int hub_velocity_index[4][3]={{" + "},{".join(
            ",".join(str(ti[f"{corner}.hub_velocity_{axis}"]) for axis in "xyz")
            for corner in FIXTURE_CORNERS) + "}};",
        "    tire_in[hub_position_index[corner][0]]=hub_x;tire_in[hub_position_index[corner][1]]=hub_y;tire_in[hub_position_index[corner][2]]=hub_z;",
        "    for(int world=0;world<3;++world)tire_in[hub_velocity_index[corner][world]]=hub_velocity[world];",
        "    const int hub_basis_index[4][3][3]={{" + "},{".join(
            "{" + "},{".join(
                ",".join(str(ti[f"{corner}.hub_basis_{local}_{world}"]) for world in "xyz")
                for local in "xyz"
            ) + "}"
            for corner in FIXTURE_CORNERS
        ) + "}};",
        "    const int hub_body_omega_index[4][3]={{" + "},{".join(
            ",".join(str(ti[f"{corner}.hub_angular_velocity_{axis}"]) for axis in "xyz")
            for corner in FIXTURE_CORNERS
        ) + "}};",
        "    for(int local=0;local<3;++local)for(int world=0;world<3;++world)tire_in[hub_basis_index[corner][local][world]]=body_basis[local][world];for(int world=0;world<3;++world)tire_in[hub_body_omega_index[corner][world]]=body_omega[world];",
        "    const int hub_angle_index[4]={" + ",".join(str(ti[f"{c}.hub_angle_rad"]) for c in FIXTURE_CORNERS) + "};",
        "    const int hub_omega_index[4]={" + ",".join(str(ti[f"{c}.hub_angular_velocity_z"]) for c in FIXTURE_CORNERS) + "};",
        "    const int wheel_angle_index[4]={" + ",".join(str(vi[f"wheel_angle_{c}"]) for c in FIXTURE_CORNERS) + "};",
        "    const int wheel_omega_index[4]={" + ",".join(str(vi[f"wheel_omega_{c}"]) for c in FIXTURE_CORNERS) + "};",
        "    tire_in[hub_angle_index[corner]]=vehicle_in[wheel_angle_index[corner]];tire_in[hub_omega_index[corner]]=vehicle_in[wheel_omega_index[corner]];",
        "    const int terrain_mode = fixture_in[" + str(fi["surface_mode"]) + "] >= 0.5;",
        "    const int plane_count = terrain_mode ? 1 : 2;",
        "    const int plane_count_index[4]={" + ",".join(str(ti[f"{c}.plane_count"]) for c in FIXTURE_CORNERS) + "};",
        "    const int surface_kind_index[4]={" + ",".join(str(ti[f"{c}.surface_kind"]) for c in FIXTURE_CORNERS) + "};",
        "    const int cylinder_radius_index[4]={" + ",".join(str(ti[f"{c}.cylinder_radius_m"]) for c in FIXTURE_CORNERS) + "};",
        f"    tire_in[surface_kind_index[corner]]=terrain_mode?0.0:1.0;tire_in[cylinder_radius_index[corner]]={roller_radius:.17g};",
        "    tire_in[plane_count_index[corner]]=plane_count;",
        "    for (int plane_index=0; plane_index<plane_count; ++plane_index) {",
        "        double plane_point[3], plane_normal[3];",
        "        if (terrain_mode) {",
        "            vehicle_periodic_terrain_plane(hub_x,hub_z,fixture_in[" + str(fi["terrain_phase_x"]) + "],fixture_in[" + str(fi["terrain_phase_z"]) + "],fixture_in[" + str(fi["terrain_period_x"]) + "],fixture_in[" + str(fi["terrain_period_z"]) + "],fixture_in[fixture_carriage_index[corner]],plane_point,plane_normal);",
        "        } else {",
        f"            const double roller_x=vehicle_native_roller_anchor[corner][0]+(plane_index?{roller_offset:.17g}:-{roller_offset:.17g});",
        "            const double roller_y=fixture_in[fixture_carriage_index[corner]];",
        "            double dx=hub_x-roller_x,dy=hub_y-roller_y,dn=sqrt(dx*dx+dy*dy); if(dn<1e-12)dn=1e-12;",
        "            plane_normal[0]=0.0;plane_normal[1]=1.0;plane_normal[2]=0.0;",
        "            plane_point[0]=roller_x;plane_point[1]=roller_y;plane_point[2]=vehicle_native_roller_anchor[corner][1];",
        "        }",
        "        const int plane_base[4][2]={{" + "},{".join(
            ",".join(str(ti[f"{corner}.plane_{plane}_point_x"]) for plane in range(2))
            for corner in FIXTURE_CORNERS) + "}};",
        "        int base=plane_base[corner][plane_index];for(int axis=0;axis<3;++axis){tire_in[base+axis]=plane_point[axis];tire_in[base+3+axis]=plane_normal[axis];tire_in[base+6+axis]=(axis==0&&terrain_mode)?fixture_in[" + str(fi["terrain_velocity_x"]) + "]:(axis==2&&terrain_mode)?fixture_in[" + str(fi["terrain_velocity_z"]) + "]:(axis==1)?fixture_in[fixture_carriage_velocity_index[corner]]:0.0;}",
        "    }",
        "}",
        "double tire_assembly_sum=vehicle_native_tire_assembly_alpha[0]+vehicle_native_tire_assembly_alpha[1]+vehicle_native_tire_assembly_alpha[2]+vehicle_native_tire_assembly_alpha[3];",
        f"if(tire_assembly_sum>0.0){{if(!tire_initialized){{balloon_tire_appendage_initialize(tire_in,tire_state);tire_initialized=1;}}double outer_dt=vehicle_in[{vi.get('dt', 0)}],wrench_sum[24]={{0}},contact_peak[4]={{0}},minimum_skin[4]={{1e300,1e300,1e300,1e300}},current_plane[4][2][3],surface_velocity[4][2][3],current_hub[4][3],current_basis[4][3][3],current_angle[4];tire_in[{ti['dt']}]=outer_dt/{tire_microsteps}.0;const int micro_plane_base[4][2]={plane_point_indices},micro_hub_base[4][3]={hub_position_indices},micro_basis_base[4][3][3]={hub_basis_indices},micro_angle_base[4]={hub_angle_indices};for(int w=0;w<4;++w){{current_angle[w]=tire_in[micro_angle_base[w]];if(!tire_pose_previous_valid)tire_angle_previous[w]=current_angle[w];for(int a=0;a<3;++a){{current_hub[w][a]=tire_in[micro_hub_base[w][a]];if(!tire_pose_previous_valid)tire_hub_previous[w][a]=current_hub[w][a];for(int b=0;b<3;++b){{current_basis[w][a][b]=tire_in[micro_basis_base[w][a][b]];if(!tire_pose_previous_valid)tire_basis_previous[w][a][b]=current_basis[w][a][b];}}}}for(int p=0;p<2;++p)for(int a=0;a<3;++a){{int b=micro_plane_base[w][p];current_plane[w][p][a]=tire_in[b+a];surface_velocity[w][p][a]=tire_in[b+6+a];if(!tire_plane_previous_valid)tire_plane_previous[w][p][a]=current_plane[w][p][a];}}}}tire_plane_previous_valid=1;tire_pose_previous_valid=1;for(int micro=0;micro<{tire_microsteps};++micro){{double alpha=(micro+1)/{tire_microsteps}.0;for(int w=0;w<4;++w){{tire_in[micro_angle_base[w]]=tire_angle_previous[w]+alpha*(current_angle[w]-tire_angle_previous[w]);for(int a=0;a<3;++a){{tire_in[micro_hub_base[w][a]]=tire_hub_previous[w][a]+alpha*(current_hub[w][a]-tire_hub_previous[w][a]);for(int b=0;b<3;++b)tire_in[micro_basis_base[w][a][b]]=tire_basis_previous[w][a][b]+alpha*(current_basis[w][a][b]-tire_basis_previous[w][a][b]);}}for(int p=0;p<2;++p)for(int a=0;a<3;++a){{int b=micro_plane_base[w][p];double geometric_velocity=(current_plane[w][p][a]-tire_plane_previous[w][p][a])/outer_dt;tire_in[b+a]=tire_plane_previous[w][p][a]+alpha*(current_plane[w][p][a]-tire_plane_previous[w][p][a]);tire_in[b+6+a]=surface_velocity[w][p][a]+geometric_velocity;}}}}balloon_tire_appendage_step(tire_in,tire_state,tire_out);for(int w=0;w<4;++w){{for(int a=0;a<6;++a)wrench_sum[6*w+a]+=tire_out[{tire_output_stride}*w+a];contact_peak[w]=fmax(contact_peak[w],tire_out[{tire_output_stride}*w+9]);minimum_skin[w]=fmin(minimum_skin[w],tire_out[{tire_output_stride}*w+10]);}}}}for(int w=0;w<4;++w){{tire_angle_previous[w]=current_angle[w];tire_in[micro_angle_base[w]]=current_angle[w];for(int a=0;a<3;++a){{tire_hub_previous[w][a]=current_hub[w][a];tire_in[micro_hub_base[w][a]]=current_hub[w][a];for(int b=0;b<3;++b){{tire_basis_previous[w][a][b]=current_basis[w][a][b];tire_in[micro_basis_base[w][a][b]]=current_basis[w][a][b];}}}}for(int p=0;p<2;++p)for(int a=0;a<3;++a){{int b=micro_plane_base[w][p];tire_plane_previous[w][p][a]=current_plane[w][p][a];tire_in[b+a]=current_plane[w][p][a];tire_in[b+6+a]=surface_velocity[w][p][a];}}for(int a=0;a<6;++a)tire_out[{tire_output_stride}*w+a]=wrench_sum[6*w+a]/{tire_microsteps}.0;tire_out[{tire_output_stride}*w+9]=contact_peak[w];tire_out[{tire_output_stride}*w+10]=minimum_skin[w];}}}}else{{memset(tire_out,0,sizeof(tire_out));tire_plane_previous_valid=0;tire_pose_previous_valid=0;}}",
        "for(int corner=0;corner<4;++corner){",
        f"    double *cin=contact_in+corner*{len(contact.input_names)};int oo=corner*{tire_output_stride};",
        "    const int assembly_alpha_index[4]={" + ",".join(str(vi[f"assembly_alpha_{c}"]) for c in FIXTURE_CORNERS) + "};",
        "    double present=vehicle_in[assembly_alpha_index[corner]]*vehicle_native_tire_assembly_alpha[corner],wheel_fx=present*tire_out[oo],roller_fy=present*tire_out[oo+1],wheel_fy=roller_fy+pillar_force_y[corner],wheel_fz=present*tire_out[oo+2];double roller_load=fmax(0.0,roller_fy),wheel_load=fmax(0.0,wheel_fy),residual_y=wheel_fy-wheel_load;",
        f"    cin[{ci['support']}]=tire_out[oo+9]>0.0;cin[{ci['tire_radial_compression']}]=0.0;cin[{ci['tire_radial_velocity']}]=0.0;",
    ]
    for index, corner in enumerate(FIXTURE_CORNERS):
        prefix = "if" if index == 0 else "else if"
        longitudinal_write = (
            f"vehicle_in[{vi[f'longitudinal_force_{corner}']}]=wheel_fx*cin[{ci['forward_x']}]+wheel_fy*cin[{ci['forward_y']}]+wheel_fz*cin[{ci['forward_z']}];"
            if f"longitudinal_force_{corner}" in vi else ""
        )
        lines.extend((
            f"    {prefix}(corner=={index}){{vehicle_in[{vi[f'contact_normal_force_{corner}']}]=wheel_load;{longitudinal_write}vehicle_in[{vi[f'tire_reaction_torque_{corner}']}]=-present*tire_out[oo+5];fixture_in[{fi[f'roller_reaction_{corner}']}]=roller_load;}}",
        ))
    lines.extend((
        "    /* Normal support acts on the unsprung node and reaches the chassis\n"
        "       through the spring graph.  Only the remaining contact wrench is\n"
        "       admitted here; adding wheel_fy again would double support. */",
        "    force[0]+=wheel_fx;force[1]+=residual_y;force[2]+=wheel_fz;",
        f"    double ax=cin[{ci['attachment_x']}],ay=cin[{ci['attachment_y']}],az=cin[{ci['attachment_z']}];",
        # Axial rim moment already enters the canonical wheel/drivetrain graph
        # as tire_reaction_torque. The contact wrench retains only bending
        # moments, avoiding a second application of the same bearing reaction.
        "    torque[0]+=ay*wheel_fz-az*residual_y+present*tire_out[oo+3];torque[1]+=az*wheel_fx-ax*wheel_fz+present*tire_out[oo+4];torque[2]+=ax*residual_y-ay*wheel_fx;",
        "}",
        f"{fixture.name}(fixture_in, fixture_out);",
    ))
    # The fixture is a stateful compiled graph. Keep its carriage state in the
    # caller-owned ABI buffer exactly as the vehicle host feeds back *_next.
    for corner in FIXTURE_CORNERS:
        position_in = f"carriage_y_{corner}"
        position_out = f"carriage_y_{corner}_next"
        velocity_in = f"carriage_velocity_y_{corner}"
        velocity_out = f"carriage_velocity_y_{corner}_next"
        if position_in in fi and position_out in fo:
            lines.append(
                f"fixture_in[{fi[position_in]}] = fixture_out[{fo[position_out]}];"
            )
        if velocity_in in fi and velocity_out in fo:
            lines.append(
                f"fixture_in[{fi[velocity_in]}] = fixture_out[{fo[velocity_out]}];"
            )
    for phase in ("terrain_phase_x", "terrain_phase_z"):
        next_phase = f"{phase}_next"
        if phase in fi and next_phase in fo:
            lines.append(
                f"fixture_in[{fi[phase]}] = fixture_out[{fo[next_phase]}];"
            )
    # Fixture forces are valid after its call; close them into both the scalar
    # wheel load lanes and the chassis wrench.
    for index, corner in enumerate(FIXTURE_CORNERS):
        lines.append(
            f"vehicle_in[{vi[f'contact_normal_force_{corner}']}] += fixture_out[{fo[f'fixture_hub_force_{corner}']}];"
        )
    for axis, offset in zip("xyz", range(3)):
        lines.append(f"vehicle_in[{vi[f'contact_wrench_force_{axis}']}] = force[{offset}];")
        lines.append(f"vehicle_in[{vi[f'contact_wrench_torque_{axis}']}] = torque[{offset}];")
    lines.append(f"{vehicle.name}(vehicle_in, vehicle_out);")
    # The shell is emitted through the repository's control-source backend;
    # it does not become a Python coordinator or a second physics integrator.
    shell_name = (
        "vehicle_native_graph_tick_dd"
        if vehicle.name.endswith("_dd") else "vehicle_native_graph_tick"
    )
    body = render_c_shell(
        ControlProgram(StatementBlock(tuple(lines))), (),
        function_name=shell_name,
        parameters=("double *vehicle_in", "double *contact_in",
                    "double *fixture_in", "double *vehicle_out"),
    )
    body = body.replace(
        f"void {shell_name}(",
        f"void {shell_name}(",
        1,
    )
    body = (
        "#if 0\n"
        "/* Scalar tick alternative disabled: the validator exports and "
        "calls only the canonical vectorized Python graph. */\n"
        + body
        + "\n#endif"
    )
    rig_point_source = f'''
#define VEHICLE_NATIVE_RIG_POINTS 16
typedef struct {{int active,mode;double local[3],target[3],target_velocity[3],command_force[3],stiffness[3],damping[3],maximum_force,reaction[6];}} VehicleNativeRigPoint;
static VehicleNativeRigPoint vehicle_native_rig_points[VEHICLE_NATIVE_RIG_POINTS];
static double vehicle_native_tire_assembly_alpha[4]={{1.0,1.0,1.0,1.0}};
static double vehicle_native_pillar_alpha[4]={{0.0,0.0,0.0,0.0}},vehicle_native_pillar_pose[4][3],vehicle_native_pillar_reaction_y[4];
static double vehicle_native_tire_gas_charge=1.0;
 static double vehicle_native_roller_anchor[4][2];
 static int vehicle_native_roller_anchor_valid[4];
TURING_EXPORT void vehicle_native_set_tire_assembly(int corner,double alpha){{if(corner>=0&&corner<4)vehicle_native_tire_assembly_alpha[corner]=fmin(1.0,fmax(0.0,alpha));}}
TURING_EXPORT void vehicle_native_set_tire_gas_charge(double fraction){{vehicle_native_tire_gas_charge=fmax(0.0,fraction);}}
TURING_EXPORT void vehicle_native_set_pillar_hub_pose(int corner,double alpha,const double *pose){{if(corner<0||corner>=4)return;vehicle_native_pillar_alpha[corner]=fmin(1.0,fmax(0.0,alpha));for(int a=0;a<3;++a)vehicle_native_pillar_pose[corner][a]=pose[a];}}
TURING_EXPORT void vehicle_native_pillar_reactions(double *out){{for(int corner=0;corner<4;++corner)out[corner]=vehicle_native_pillar_reaction_y[corner];}}
TURING_EXPORT void vehicle_native_set_roller_anchor(int corner,double x,double z){{if(corner<0||corner>=4)return;vehicle_native_roller_anchor[corner][0]=x;vehicle_native_roller_anchor[corner][1]=z;vehicle_native_roller_anchor_valid[corner]=1;}}
TURING_EXPORT void vehicle_native_rig_point_clear(int slot){{if(slot>=0&&slot<VEHICLE_NATIVE_RIG_POINTS)memset(&vehicle_native_rig_points[slot],0,sizeof(VehicleNativeRigPoint));}}
TURING_EXPORT void vehicle_native_rig_point_configure(int slot,int mode,const double *v){{if(slot<0||slot>=VEHICLE_NATIVE_RIG_POINTS)return;VehicleNativeRigPoint *p=&vehicle_native_rig_points[slot];memset(p,0,sizeof(*p));p->active=1;p->mode=mode;for(int a=0;a<3;++a){{p->local[a]=v[a];p->target[a]=v[3+a];p->target_velocity[a]=v[6+a];p->command_force[a]=v[9+a];p->stiffness[a]=v[12+a];p->damping[a]=v[15+a];}}p->maximum_force=fmax(0.0,v[18]);}}
TURING_EXPORT void vehicle_native_rig_point_reactions(double *out){{for(int p=0;p<VEHICLE_NATIVE_RIG_POINTS;++p)for(int a=0;a<6;++a)out[6*p+a]=vehicle_native_rig_points[p].reaction[a];}}
static void vehicle_native_apply_rig_points(const double *in,double *force,double *torque){{
 const double cr=cos(in[{vi['roll']}]),sr=sin(in[{vi['roll']}]),cp=cos(in[{vi['pitch']}]),sp=sin(in[{vi['pitch']}]),cy=cos(in[{vi['yaw']}]),sy=sin(in[{vi['yaw']}]);
 const double omega[3]={{in[{vi['roll_velocity']}],-in[{vi['yaw_velocity']}],in[{vi['pitch_velocity']}]}},body_v[3]={{in[{vi['velocity_x']}],in[{vi['velocity_y']}],in[{vi['velocity_z']}]}},body_p[3]={{in[{vi['position_x']}],in[{vi['position_y']}],in[{vi['position_z']}]}};
 for(int i=0;i<VEHICLE_NATIVE_RIG_POINTS;++i){{VehicleNativeRigPoint *p=&vehicle_native_rig_points[i];if(!p->active)continue;double x1=cy*p->local[0]-sy*p->local[2],z1=sy*p->local[0]+cy*p->local[2],y1=cr*p->local[1]-sr*z1,r[3]={{cp*x1-sp*y1,sp*x1+cp*y1,sr*p->local[1]+cr*z1}},pv[3]={{body_v[0]+omega[1]*r[2]-omega[2]*r[1],body_v[1]+omega[2]*r[0]-omega[0]*r[2],body_v[2]+omega[0]*r[1]-omega[1]*r[0]}},f[3]={{0,0,0}},q=0;
  for(int a=0;a<3;++a){{if(p->mode==1)f[a]=p->stiffness[a]*(p->target[a]-(body_p[a]+r[a]))+p->damping[a]*(p->target_velocity[a]-pv[a]);else if(p->mode==2)f[a]=p->command_force[a];else if(p->mode==3)f[a]=p->damping[a]*(p->target_velocity[a]-pv[a]);q+=f[a]*f[a];}}
  q=sqrt(q);if(p->maximum_force>0&&q>p->maximum_force)for(int a=0;a<3;++a)f[a]*=p->maximum_force/q;
  double m[3]={{r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]}};for(int a=0;a<3;++a){{force[a]+=f[a];torque[a]+=(a==1?-m[a]:m[a]);p->reaction[a]=-f[a];p->reaction[3+a]=-m[a];}}
 }}
}}
'''
    batch_source = f'''
/* A compiler-emitted leading batch axis over the exact scalar graph.  Each
   lane owns every persistent balloon/contact history value; the map never
   averages physics state or feeds ensemble presentation back into a lane. */
#define VEHICLE_NATIVE_BATCH_CAPACITY 8
TURING_EXPORT void vehicle_graph_tick_vector(double*,double*,double*,double*);
typedef struct {{
 double tire_in[{len(tire.input_names)}],tire_state[{tire.state_scalar_count}],tire_out[{len(tire.output_names)}];
 double plane_previous[4][2][3],hub_previous[4][3],basis_previous[4][3][3],angle_previous[4];
 double roller_anchor[4][2];
 int roller_anchor_valid[4],defaults_loaded,initialized,plane_previous_valid,pose_previous_valid;
}} VehicleNativeBatchLane;
static VehicleNativeBatchLane vehicle_native_batch_lane[VEHICLE_NATIVE_BATCH_CAPACITY];
static void vehicle_native_batch_save(VehicleNativeBatchLane *q){{
 memcpy(q->tire_in,tire_in,sizeof(tire_in));memcpy(q->tire_state,tire_state,sizeof(tire_state));memcpy(q->tire_out,tire_out,sizeof(tire_out));
 memcpy(q->plane_previous,tire_plane_previous,sizeof(tire_plane_previous));memcpy(q->hub_previous,tire_hub_previous,sizeof(tire_hub_previous));memcpy(q->basis_previous,tire_basis_previous,sizeof(tire_basis_previous));memcpy(q->angle_previous,tire_angle_previous,sizeof(tire_angle_previous));
 memcpy(q->roller_anchor,vehicle_native_roller_anchor,sizeof(vehicle_native_roller_anchor));memcpy(q->roller_anchor_valid,vehicle_native_roller_anchor_valid,sizeof(vehicle_native_roller_anchor_valid));
 q->defaults_loaded=tire_defaults_loaded;q->initialized=tire_initialized;q->plane_previous_valid=tire_plane_previous_valid;q->pose_previous_valid=tire_pose_previous_valid;
}}
static void vehicle_native_batch_load(const VehicleNativeBatchLane *q){{
 memcpy(tire_in,q->tire_in,sizeof(tire_in));memcpy(tire_state,q->tire_state,sizeof(tire_state));memcpy(tire_out,q->tire_out,sizeof(tire_out));
 memcpy(tire_plane_previous,q->plane_previous,sizeof(tire_plane_previous));memcpy(tire_hub_previous,q->hub_previous,sizeof(tire_hub_previous));memcpy(tire_basis_previous,q->basis_previous,sizeof(tire_basis_previous));memcpy(tire_angle_previous,q->angle_previous,sizeof(tire_angle_previous));
 memcpy(vehicle_native_roller_anchor,q->roller_anchor,sizeof(vehicle_native_roller_anchor));memcpy(vehicle_native_roller_anchor_valid,q->roller_anchor_valid,sizeof(vehicle_native_roller_anchor_valid));
 tire_defaults_loaded=q->defaults_loaded;tire_initialized=q->initialized;tire_plane_previous_valid=q->plane_previous_valid;tire_pose_previous_valid=q->pose_previous_valid;
}}
TURING_EXPORT void vehicle_native_graph_batch_reset(void){{memset(vehicle_native_batch_lane,0,sizeof(vehicle_native_batch_lane));}}
#if 0
/* Iterative scalar batch wrapper disabled.  The canonical vectorized Python
   graph is the only exported tick implementation. */
int vehicle_native_graph_tick_batch(int count,double *vehicle_batch,double *contact_batch,double *fixture_batch,double *output_batch){{
 VehicleNativeBatchLane scalar_lane;if(count<0)count=0;if(count>VEHICLE_NATIVE_BATCH_CAPACITY)count=VEHICLE_NATIVE_BATCH_CAPACITY;vehicle_native_batch_save(&scalar_lane);
 for(int lane=0;lane<count;++lane){{vehicle_native_batch_load(&vehicle_native_batch_lane[lane]);vehicle_graph_tick_vector(vehicle_batch+lane*{len(vehicle.input_names)},contact_batch+lane*{4 * len(contact.input_names)},fixture_batch+lane*{len(fixture.input_names)},output_batch+lane*{len(vehicle.output_names)});vehicle_native_batch_save(&vehicle_native_batch_lane[lane]);}}
 vehicle_native_batch_load(&scalar_lane);return count;
}}
#endif
TURING_EXPORT void vehicle_native_graph_batch_tire_state(int count,double *out){{if(count<0)count=0;if(count>VEHICLE_NATIVE_BATCH_CAPACITY)count=VEHICLE_NATIVE_BATCH_CAPACITY;for(int lane=0;lane<count;++lane)memcpy(out+lane*{tire.state_scalar_count},vehicle_native_batch_lane[lane].tire_state,sizeof(vehicle_native_batch_lane[lane].tire_state));}}
'''
    return "\n".join((
        "#include <math.h>",
        "#include <string.h>",
        "#if defined(_WIN32)",
        "#define TURING_EXPORT __declspec(dllexport)",
        "#else",
        "#define TURING_EXPORT __attribute__((visibility(\"default\")))",
        "#endif",
        rig_point_source,
        material_source,
        f"void {vehicle.name}(const double *, double *);",
        f"void {fixture.name}(const double *, double *);",
        "void balloon_tire_appendage_defaults(double *);",
        "void balloon_tire_appendage_initialize(const double *,double *);",
        "void balloon_tire_appendage_step(const double *,double *,double *);",
        "void balloon_tire_contact_diagnostics(double *);",
        f"static double tire_in[{len(tire.input_names)}],tire_state[{tire.state_scalar_count}],tire_out[{len(tire.output_names)}];",
        "static int tire_defaults_loaded=0,tire_initialized=0,tire_plane_previous_valid=0,tire_pose_previous_valid=0;",
        "static double tire_plane_previous[4][2][3];",
        "static double tire_hub_previous[4][3],tire_basis_previous[4][3][3],tire_angle_previous[4];",
        "static double vehicle_appendage_duty_state[4]={0.0,1.0,1.0,1.0};",
        "TURING_EXPORT void vehicle_native_set_appendage_duty(double learned,double exact_trial,double exact_alpha,double sampled_pulse){vehicle_appendage_duty_state[0]=learned;vehicle_appendage_duty_state[1]=exact_trial;vehicle_appendage_duty_state[2]=exact_alpha;vehicle_appendage_duty_state[3]=sampled_pulse;}",
        "TURING_EXPORT void vehicle_native_appendage_duty(double *out){for(int i=0;i<4;++i)out[i]=vehicle_appendage_duty_state[i];}",
        f"TURING_EXPORT void vehicle_native_tire_diagnostics(double *out){{for(int i=0;i<{len(tire.output_names)};++i)out[i]=tire_out[i];balloon_tire_contact_diagnostics(out+{len(tire.output_names)});}}",
        f"TURING_EXPORT void vehicle_native_tire_state(double *out){{for(int i=0;i<{tire.state_scalar_count};++i)out[i]=tire_state[i];}}",
        f"TURING_EXPORT void vehicle_native_restore_tire_state(const double *in){{for(int i=0;i<{tire.state_scalar_count};++i)tire_state[i]=in[i];tire_initialized=1;tire_plane_previous_valid=0;tire_pose_previous_valid=0;}}",
        f"TURING_EXPORT void vehicle_native_energy_diagnostics(double *out){{double kinetic=0.0;for(int w=0;w<4;++w)for(int v=0;v<{tire.vertex_count};++v){{double *s=tire_state+w*{tire.vertex_count * 6}+6*v;kinetic+=0.5*tire_in[{ti['vertex_mass_kg']}]*(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]);}}out[0]=kinetic;out[1]=out[2]=out[3]=0.0;for(int w=0;w<4;++w){{out[1]+=tire_out[{tire_output_stride}*w+11]+tire_out[{tire_output_stride}*w+{bending_energy_offset}];out[2]+=tire_out[{tire_output_stride}*w+12];out[3]+=tire_out[{tire_output_stride}*w+9];}}}}",
        f"TURING_EXPORT void vehicle_native_reset(void){{memset(tire_in,0,sizeof(tire_in));memset(tire_state,0,sizeof(tire_state));memset(tire_out,0,sizeof(tire_out));memset(vehicle_native_material_state,0,sizeof(vehicle_native_material_state));memset(vehicle_native_material_diagnostic,0,sizeof(vehicle_native_material_diagnostic));memset(tire_plane_previous,0,sizeof(tire_plane_previous));memset(tire_hub_previous,0,sizeof(tire_hub_previous));memset(tire_basis_previous,0,sizeof(tire_basis_previous));memset(tire_angle_previous,0,sizeof(tire_angle_previous));memset(vehicle_native_rig_points,0,sizeof(vehicle_native_rig_points));memset(vehicle_native_pillar_pose,0,sizeof(vehicle_native_pillar_pose));memset(vehicle_native_pillar_reaction_y,0,sizeof(vehicle_native_pillar_reaction_y));memset(vehicle_native_roller_anchor,0,sizeof(vehicle_native_roller_anchor));memset(vehicle_native_roller_anchor_valid,0,sizeof(vehicle_native_roller_anchor_valid));for(int i=0;i<4;++i){{vehicle_native_tire_assembly_alpha[i]=1.0;vehicle_native_pillar_alpha[i]=0.0;}}vehicle_native_tire_gas_charge=1.0;tire_defaults_loaded=0;tire_initialized=0;tire_plane_previous_valid=0;tire_pose_previous_valid=0;}}",
        "static double vehicle_terrain_vertex(int ix,int iz,int nx,int nz){double u=6.2831853071795864769*(double)((ix%nx+nx)%nx)/(double)nx,v=6.2831853071795864769*(double)((iz%nz+nz)%nz)/(double)nz;return .075*sin(u)+.045*sin(v*2.0+.7)+.035*sin(u+v*1.5);}",
        "static void vehicle_periodic_terrain_plane(double x,double z,double phase_x,double phase_z,double period_x,double period_z,double base_y,double *p,double *n){const int nx=8,nz=8;double sx=period_x/nx,sz=period_z/nz,u=fmod(x+phase_x,period_x),v=fmod(z+phase_z,period_z);if(u<0)u+=period_x;if(v<0)v+=period_z;int ix=(int)floor(u/sx),iz=(int)floor(v/sz);double fu=(u-ix*sx)/sx,fv=(v-iz*sz)/sz;double ax,az,bx,bz,cx,cz,ay,by,cy;if(fu+fv<=1.0){ax=ix*sx;az=iz*sz;ay=vehicle_terrain_vertex(ix,iz,nx,nz);bx=(ix+1)*sx;bz=iz*sz;by=vehicle_terrain_vertex(ix+1,iz,nx,nz);cx=ix*sx;cz=(iz+1)*sz;cy=vehicle_terrain_vertex(ix,iz+1,nx,nz);}else{ax=(ix+1)*sx;az=(iz+1)*sz;ay=vehicle_terrain_vertex(ix+1,iz+1,nx,nz);bx=ix*sx;bz=(iz+1)*sz;by=vehicle_terrain_vertex(ix,iz+1,nx,nz);cx=(ix+1)*sx;cz=iz*sz;cy=vehicle_terrain_vertex(ix+1,iz,nx,nz);}double e1x=bx-ax,e1y=by-ay,e1z=bz-az,e2x=cx-ax,e2y=cy-ay,e2z=cz-az;n[0]=e1z*e2y-e1y*e2z;n[1]=e1x*e2z-e1z*e2x;n[2]=e1y*e2x-e1x*e2y;double q=sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);if(n[1]<0)q=-q;n[0]/=q;n[1]/=q;n[2]/=q;p[0]=x;p[2]=z;p[1]=base_y+ay-(n[0]*(u-ax)+n[2]*(v-az))/n[1];}",
        body,
        batch_source,
    ))


def scientific_visualization_abi() -> Mapping[str, Any]:
    """Describe the read-only vertex contract consumed by diagnostic GLSL."""

    return {
        "authority": "read-only compiler-output consumer",
        "attributes": [
            {"location": 0, "name": "position", "type": "vec3", "units": "m"},
            {"location": 1, "name": "normal", "type": "vec3", "units": "1"},
            {"location": 2, "name": "component_type", "type": "float-coded-int"},
            {"location": 3, "name": "axial_strain", "type": "float", "units": "m/m"},
            {"location": 4, "name": "yield_strain", "type": "float", "units": "m/m"},
            {"location": 5, "name": "fracture_strain", "type": "float", "units": "m/m"},
            {"location": 6, "name": "damage_state", "type": "float-coded-int"},
            {"location": 7, "name": "joint_kind", "type": "float-coded-int"},
            {"location": 8, "name": "laplacian_energy", "type": "float", "units": "log-scaled J"},
            {"location": 9, "name": "conservation_residual", "type": "float", "units": "log-scaled W"},
        ],
        "component_types": {
            "0": "frame-structure", "1": "suspension-link", "2": "drivetrain-rotating",
            "3": "tire-contact", "4": "external-fixture", "5": "hydraulic-pneumatic",
            "6": "electrical-control", "7": "body-other",
        },
        "joint_kinds": {"0": "member", "1": "bushing-dot", "2": "bearing-dot"},
        "strain_color": {
            "zero": "component-type base hue",
            "compression": "blue diverging tint",
            "tension": "red diverging tint",
            "plastic": "magenta overlay",
            "fractured": "white-black diagnostic hatch",
        },
        "physics_effect": "none",
    }


def _c_initializers(names: tuple[str, ...], values: Mapping[str, float]) -> str:
    return ",".join(format(float(values.get(name, 0.0)), ".17g") for name in names)


def render_native_scientific_viewer_shell(
    vehicle: CFunctionArtifact,
    contact: CFunctionArtifact,
    fixture: CFunctionArtifact,
    *,
    target_rate_hz: int = 120,
    assembly_profile: Any | None = None,
) -> str:
    """Transport the compiled vehicle ABI into the standard native GL shell."""

    config = load_default_car_configuration()
    physics_hz = int(target_rate_hz)
    if physics_hz <= 0:
        raise ValueError("scientific viewer target rate must be positive")
    graph = (
        _vehicle_mechanical_graph(config)
        if assembly_profile is None
        else assembly_profile.model["mechanical_graph"]
    )
    defaults = {name: 0.0 for name in vehicle.input_names}
    defaults.update(config.parameter_defaults())
    defaults.update({
        "dt": 1.0 / physics_hz, "position_y": 0.9, "yaw_cos": 1.0,
        "forward_gear_ratio": 1.0, "transfer_case_ratio": 1.0,
        "drive_fraction_front_left": 0.21, "drive_fraction_front_right": 0.21,
        "drive_fraction_rear_left": 0.29, "drive_fraction_rear_right": 0.29,
    })
    if assembly_profile is not None:
        mass = assembly_profile.mass_properties
        defaults.update({
            "position_x": 0.0, "position_y": 0.0, "position_z": 0.0,
            "gravity": 0.0,
            "inverse_mass": 1.0 / float(mass["mass_kg"]),
            "inverse_inertia_roll": 1.0 / float(
                mass["inertia_kg_m2"]["roll"]),
            "inverse_inertia_pitch": 1.0 / float(
                mass["inertia_kg_m2"]["pitch"]),
            "inverse_inertia_yaw": 1.0 / float(
                mass["inertia_kg_m2"]["yaw"]),
            "assembly_alpha_drivetrain": 0.0,
            **{
                f"assembly_alpha_{corner}": 0.0
                for corner in FIXTURE_CORNERS
            },
        })
    contact_defaults = {name: 0.0 for name in contact.input_names}
    contact_defaults.update(defaults)
    tire = config.source["tires"]
    if assembly_profile is None:
        tire_radius = float(tire["radius"])
        section_radius = float(tire["toroid_section_radius_m"])
        tire_width = float(tire["width"])
        tire_pressure = float(tire["pressure_pa"])
        tire_mass = config.unsprung_mass_per_corner()
    else:
        (tire_radius, section_radius, tire_width, tire_mass,
         tire_pressure, _rim_radius) = assembly_profile.tire_dimensions
    major_radius = tire_radius - section_radius
    contact_defaults.update({
        "support": 1.0, "normal_y": 1.0, "forward_x": 1.0, "right_z": 1.0,
        "corner_weight": float(defaults.get("mass", 620.0)) * 9.81 / 4.0,
        "load_sensitivity": 0.075, "maximum_contact_area": 0.06,
        "minimum_contact_area": 0.008, "mu_kinetic": 0.92, "mu_static": 1.18,
        "slip_transition_speed": float(tire["slip_transition_speed"]),
        "radial_carcass_loss": float(tire["radial_carcass_loss_n_s_per_m"]),
        "tire_radial_effective_mass": tire_mass
        * float(tire["radial_contact_effective_mass_fraction_of_unsprung"]),
        "sidewall_shear_damping": float(tire["sidewall_shear_damping_n_s_per_m"]),
        "sidewall_shear_stiffness_lateral": float(tire["sidewall_shear_stiffness_lateral_n_per_m"]),
        "sidewall_shear_stiffness_longitudinal": float(tire["sidewall_shear_stiffness_longitudinal_n_per_m"]),
        "tire_effective_tread_width": tire_width * float(tire["effective_tread_width_fraction"]),
        "tire_gas_polytropic_exponent": float(tire["gas_polytropic_exponent"]),
        "tire_major_radius": major_radius,
        "tire_pressure": tire_pressure,
        "tire_radial_compression": 0.035,
        "tire_reference_volume": 2.0 * float(sympy.pi) ** 2 * major_radius * section_radius ** 2,
        "tire_section_radius": section_radius,
    })
    if assembly_profile is None:
        wheels, chassis = config.source["wheels"], config.source["chassis"]
        wheelbase = float(wheels["wheelbase_half_length"])
        axle_offset = float(wheels["axle_group_offset_x_m"])
        track = float(wheels["track_half_width"])
        attachment_y = -float(chassis["clearance"])
        attachments = (
            (axle_offset + wheelbase, attachment_y, -track),
            (axle_offset + wheelbase, attachment_y, track),
            (axle_offset - wheelbase, attachment_y, -track),
            (axle_offset - wheelbase, attachment_y, track),
        )
    else:
        attachments = assembly_profile.wheel_attachments
    contact_values: list[float] = []
    for attachment in attachments:
        one = dict(contact_defaults)
        one.update(dict(zip(("attachment_x", "attachment_y", "attachment_z"), attachment)))
        contact_values.extend(float(one.get(name, 0.0)) for name in contact.input_names)
    fixture_defaults = {name: 0.0 for name in fixture.input_names}
    fixture_defaults.update({
        "dt": 1.0 / physics_hz, "gravity": -9.81, "floor_y": -0.75,
        "carriage_mass": 12.0, "neutral_buoyancy": 1.0, "passive_damping": 8.0,
        "lock_stiffness": 24_000.0, "lock_damping": 1_200.0,
        "maximum_actuator_force": 18_000.0, "mode": float(HOLDER_MODES["cage-drive"]),
        "surface_mode": 0.0, "terrain_phase_x": 0.0, "terrain_phase_z": 0.0,
        "terrain_velocity_x": 1.2, "terrain_velocity_z": 0.35,
        "terrain_period_x": 4.0, "terrain_period_z": 4.0,
    })
    initial_hub_y = (
        float(config.source["suspension"]["assembly_hub_height_m"])
        if assembly_profile is None else float(attachments[0][1])
    )
    # Spawn at geometric tangency. The former 35 mm preload initialized the
    # membrane behind the hard surface and poisoned the first CCD step with a
    # non-crossing correction impulse.
    initial_radial_compression = 0.0
    roller_center_distance = (
        tire_radius + 0.13
        - initial_radial_compression
    )
    initial_vertical_separation = max(
        roller_center_distance ** 2 - 0.18 ** 2, 0.0,
    ) ** 0.5
    initial_carriage_y = initial_hub_y - initial_vertical_separation
    for corner in FIXTURE_CORNERS:
        fixture_defaults[f"carriage_y_{corner}"] = initial_carriage_y
        fixture_defaults[f"command_y_{corner}"] = initial_carriage_y
        fixture_defaults[f"hub_y_{corner}"] = initial_hub_y
        fixture_defaults[f"mode_{corner}"] = float(HOLDER_MODES["cage-drive"])

    vi = {name: index for index, name in enumerate(vehicle.input_names)}
    vo = {name: index for index, name in enumerate(vehicle.output_names)}
    fi = {name: index for index, name in enumerate(fixture.input_names)}
    feedback = []
    batch_feedback = []
    for output_name, output_index in vo.items():
        if output_name.endswith("_next") and output_name[:-5] in vi:
            feedback.append(
                f"vehicle_in[{vi[output_name[:-5]]}]=vehicle_out[{output_index}];"
            )
            batch_feedback.append(
                f"bvi[{vi[output_name[:-5]]}]=bvo[{output_index}];"
            )
    template_path = Path(__file__).with_name("vehicle_scientific_viewer_shell.c.in")
    template = template_path.read_text(encoding="utf-8")

    def component_type(*labels: str) -> int:
        text = " ".join(labels).casefold()
        if any(word in text for word in ("wire", "electric", "ecu", "lamp", "battery")):
            return 6
        if any(word in text for word in ("hydraulic", "pneumatic", "hose", "outrigger")):
            return 5
        if any(word in text for word in ("tire", "tyre", "contact-patch", "wheel-rim")):
            return 3
        if any(word in text for word in ("drivetrain", "shaft", "bearing", "rotor", "brake", "hub")):
            return 2
        if any(word in text for word in ("suspension", "steering", "spring", "damper", "arm", "coilover", "knuckle")):
            return 1
        if any(word in text for word in ("body", "shell", "turret", "weapon", "armor")):
            return 7
        return 0

    def c_float(value: float) -> str:
        text = format(float(value), ".9g")
        if "." not in text and "e" not in text.casefold():
            text += ".0"
        return text + "f"

    nodes = tuple(graph["nodes"])
    edges = tuple(graph["edges"])
    node_index = {node["identity"]: index for index, node in enumerate(nodes)}
    assembly_stages = native_vehicle_assembly_stages()
    assembly_stage_identities = (
        tuple(stage.identity for stage in assembly_stages)
        if assembly_profile is None
        else tuple(assembly_profile.stages)
    )

    def reveal_stage(*labels: str) -> int:
        """Map visible graph material to the stage that physically installs it."""

        text = " ".join(labels).casefold()
        if assembly_profile is not None:
            for stage_index, identity in enumerate(assembly_stage_identities):
                if identity.casefold() in text:
                    return stage_index
            if any(word in text for word in (
                "tire", "tyre", "wheel", "rim", "bead",
            )):
                return 1
            return 0
        for stage_index, stage in enumerate(assembly_stages):
            if any(pattern.casefold() in text for pattern in stage.component_patterns):
                return stage_index
        if any(word in text for word in ("tire", "tyre", "wheel", "rim", "bead")):
            return 6
        if any(word in text for word in ("front_", "front.", "front-")) and any(
            word in text for word in ("suspension", "knuckle", "arm", "bearing", "rotor", "caliper")
        ):
            return 9
        if any(word in text for word in ("rear_", "rear.", "rear-")) and any(
            word in text for word in ("suspension", "knuckle", "arm", "bearing", "rotor", "caliper")
        ):
            return 10
        if any(word in text for word in (
            "body", "electrical", "wiring", "lamp", "hydraulic", "pneumatic",
            "fuel", "starter", "computer", "ecu", "servo",
        )):
            return 14
        return 0
    bushing_nodes: set[str] = set()
    bearing_nodes: set[str] = set()
    for edge in edges:
        endpoint_a, endpoint_b = _mechanical_edge_endpoints(edge)
        if edge.get("joint_bushings"):
            bushing_nodes.update((endpoint_a, endpoint_b))
        edge_relationship = " ".join((
            str(edge.get("constraint", "")), str(edge.get("kind", "")),
        )).casefold()
        if "bearing" in edge_relationship:
            bearing_nodes.update((endpoint_a, endpoint_b))
    corner_index = {corner: index for index, corner in enumerate(FIXTURE_CORNERS)}
    node_rows = []
    for node in nodes:
        identity = str(node["identity"])
        kind = str(node.get("kind", ""))
        corner = next((index for name, index in corner_index.items()
                       if f".{name}." in f".{identity}."), -1)
        moving = int(corner >= 0 and (
            kind != "chassis-pickup" or identity.endswith(("steering_arm", "brake_caliper"))
        ))
        joint = 2 if identity in bearing_nodes or "bearing" in kind.casefold() else (
            1 if identity in bushing_nodes else 0
        )
        x, y, z = map(float, node["reference_position"])
        node_rows.append(
            "{" + ",".join((format(x, ".9g"), format(y, ".9g"), format(z, ".9g"),
                              f"{component_type(identity, kind)}.0f", f"{joint}.0f",
                              str(corner), str(moving), str(reveal_stage(
                                  identity, kind,
                                  str(node.get("assembly_stage", "")))))) + "}"
        )
    edge_rows = []
    for edge in edges:
        damage = edge.get("damage") or {}
        endpoint_a, endpoint_b = _mechanical_edge_endpoints(edge)
        a_position = nodes[node_index[endpoint_a]]["reference_position"]
        b_position = nodes[node_index[endpoint_b]]["reference_position"]
        geometric_rest = sum(
            (float(a_position[axis]) - float(b_position[axis])) ** 2
            for axis in range(3)
        ) ** 0.5
        # The scientific view starts from this exact reference geometry.  An
        # actuator's commanded length is not the member's zero-strain display
        # length; conflating them produces enormous phantom utilization before
        # the first kernel tick.
        rest = geometric_rest
        yield_strain = max(float(damage.get("plastic_strain_limit") or 0.0025), 1.0e-6)
        fracture_strain = max(float(damage.get("fracture_strain") or 0.075), yield_strain)
        bushings = edge.get("joint_bushings") or {}
        bushing_damping = sum(
            float((bushings.get(end) or {}).get("linear_damping_n_s_per_m") or 0.0)
            for end in ("a", "b")
        )
        stiffness = float(
            damage.get("axial_stiffness_n_per_m")
            or edge.get("compression_stiffness_n_per_m")
            or edge.get("radial_stiffness_n_per_m")
            or edge.get("axial_stiffness_n_per_m")
            or edge.get("shear_stiffness_n_per_m")
            or edge.get("stiffness_n_per_m")
            or edge.get("stiffness")
            or 0.0
        )
        damping = float(
            edge.get("damping_n_s_per_m")
            or edge.get("compression_damping_n_s_per_m")
            or edge.get("compression_damping")
            or bushing_damping
            or 0.0
        )
        edge_rows.append(
            "{" + ",".join((str(node_index[endpoint_a]), str(node_index[endpoint_b]),
                              f"{component_type(str(edge.get('identity', '')), str(edge.get('constraint', '')))}.0f",
                              c_float(rest), c_float(yield_strain),
                              c_float(fracture_strain), c_float(stiffness),
                               c_float(damping), str(max(
                                   reveal_stage(str(edge.get("identity", "")), str(edge.get("constraint", ""))),
                                   reveal_stage(endpoint_a), reveal_stage(endpoint_b),
                               )))) + "}"
        )
    wheel_nodes = (
        [node_index[f"suspension.{corner}.hub"] for corner in FIXTURE_CORNERS]
        if assembly_profile is None
        else [node_index[wheel["identity"]]
              for wheel in assembly_profile.model["wheels"]]
    )
    ci = {name: index for index, name in enumerate(contact.input_names)}
    tune_parameters = [
        ("SPRING STIFFNESS", 0, vi["spring_stiffness"], 8_000.0, 120_000.0),
        ("PNEUMATIC COMP DAMP", 0, vi["pneumatic_compression_damping"], 200.0, 20_000.0),
        ("PNEUMATIC REBOUND DAMP", 0, vi["pneumatic_rebound_damping"], 200.0, 20_000.0),
        ("ANGULAR DAMPING", 0, vi["angular_damping"], 0.2, 16.0),
        ("TIRE PRESSURE", 1, ci["tire_pressure"], 60_000.0, 450_000.0),
        ("CARCASS LOSS", 1, ci["radial_carcass_loss"], 50.0, 8_000.0),
        ("SIDEWALL DAMPING", 1, ci["sidewall_shear_damping"], 20.0, 6_000.0),
        ("SIDEWALL LONG STIFF", 1, ci["sidewall_shear_stiffness_longitudinal"], 40_000.0, 900_000.0),
        ("SIDEWALL LAT STIFF", 1, ci["sidewall_shear_stiffness_lateral"], 40_000.0, 900_000.0),
    ]
    if "tire_radial_effective_mass" in ci:
        tune_parameters.insert(
            6, ("RADIAL EFFECTIVE MASS", 1, ci["tire_radial_effective_mass"], 10.0, 320.0))
    tune_rows = ",".join(
        "{" + ",".join((json.dumps(name), str(target), str(index),
                            format(lower, ".17g"), format(upper, ".17g"))) + "}"
        for name, target, index, lower, upper in tune_parameters
    )
    from .vehicle_balloon_tire import balloon_tire_graph_abi
    from .vehicle_balloon_tire_program import balloon_tire_python_program
    if assembly_profile is None:
        tire_program = balloon_tire_python_program()
        tire_topology = balloon_tire_graph_abi(config.source)["topology"]
        tire_rest_positions = tire_topology.rest_positions
        tire_edges = tire_topology.edges
    else:
        dims = assembly_profile.tire_dimensions
        tire_program = balloon_tire_python_program(
            assembly_profile.wheel_names,
            tire_radius_m=dims[0], tire_section_radius_m=dims[1],
            tire_width_m=dims[2], tire_mass_kg=dims[3],
            reference_pressure_pa=dims[4], rim_radius_m=dims[5],
            pneumatic_mode=assembly_profile.tire_pneumatic_mode,
            material_profile=assembly_profile.tire_material_profile,
        )
        tire_rest_positions = tuple(map(tuple, tire_program.constants["rest"]))
        tire_edges = tuple(sorted({
            tuple(sorted((int(face[left]), int(face[right]))))
            for face in tire_program.constants["face_vertices"]
            for left, right in ((0, 1), (1, 2), (2, 0))
        }))
    tire_output_count = len(tire_program.output_names)
    tire_edge_rows = []
    for a, b in tire_edges:
        pa, pb = tire_rest_positions[a], tire_rest_positions[b]
        rest = math.sqrt(sum((pa[axis] - pb[axis]) ** 2 for axis in range(3)))
        tire_edge_rows.append(f"{{{a},{b},{rest:.9g}f}}")
    replacements = {
        "@VEHICLE_INPUT_COUNT@": str(len(vehicle.input_names)),
        "@VEHICLE_OUTPUT_COUNT@": str(len(vehicle.output_names)),
        "@CONTACT_TOTAL_COUNT@": str(len(contact.input_names) * 4),
        "@FIXTURE_INPUT_COUNT@": str(len(fixture.input_names)),
        "@TIRE_STATE_COUNT@": str(4 * len(tire_rest_positions) * 6),
        "@TIRE_VERTEX_COUNT@": str(len(tire_rest_positions)),
        "@TIRE_EDGE_COUNT@": str(len(tire_edges)),
        "@TIRE_EDGES@": ",".join(tire_edge_rows),
        "@VEHICLE_DEFAULTS@": _c_initializers(vehicle.input_names, defaults),
        "@CONTACT_DEFAULTS@": ",".join(format(value, ".17g") for value in contact_values),
        "@FIXTURE_DEFAULTS@": _c_initializers(fixture.input_names, fixture_defaults),
        "@VEHICLE_FEEDBACK@": "".join(feedback),
        "@VEHICLE_BATCH_FEEDBACK@": "".join(batch_feedback),
        "@PHYSICS_HZ@": str(physics_hz),
        # One display iteration requests one target game window.  Interior
        # work belongs exclusively to the managed dt controller.
        "@STEPS_PER_FRAME@": "1",
        "@ENSEMBLE_STEPS_PER_FRAME@": "2",
        "@GRAPH_NODE_COUNT@": str(len(nodes)),
        "@GRAPH_EDGE_COUNT@": str(len(edges)),
        "@GRAPH_NODES@": ",".join(node_rows),
        "@GRAPH_EDGES@": ",".join(edge_rows),
        "@GRAPH_EDGE_NAMES@": ",".join(
            json.dumps(str(edge.get("identity", "member"))[:28]) for edge in edges
        ),
        "@ASSEMBLY_STAGE_COUNT@": str(len(assembly_stage_identities)),
        "@MOUNT_TIRE_STAGE@": str(next(
            index for index, identity in enumerate(assembly_stage_identities)
            if "tire-casing" in identity)),
        "@WHEEL_NODES@": ",".join(map(str, wheel_nodes)),
        "@WHEEL_RADIUS@": format(tire_radius, ".9g"),
        "@WHEEL_WIDTH@": format(tire_width, ".9g"),
        "@CONTACT_STRIDE@": str(len(contact.input_names)),
        "@CONTACT_SUPPORT@": str(ci["support"]),
        "@CONTACT_COMPRESSION@": str(ci["tire_radial_compression"]),
        "@TUNE_PARAM_COUNT@": str(len(tune_parameters)),
        "@TUNE_PARAM_ROWS@": tune_rows,
        "@TIRE_DIAGNOSTIC_COUNT@": str(tire_output_count + 20),
        "@CARRIAGE_FL@": str(fi["carriage_y_front_left"]),
        "@CARRIAGE_FR@": str(fi["carriage_y_front_right"]),
        "@CARRIAGE_RL@": str(fi["carriage_y_rear_left"]),
        "@CARRIAGE_RR@": str(fi["carriage_y_rear_right"]),
        "@COMMAND_FL@": str(fi["command_y_front_left"]),
        "@COMMAND_FR@": str(fi["command_y_front_right"]),
        "@COMMAND_RL@": str(fi["command_y_rear_left"]),
        "@COMMAND_RR@": str(fi["command_y_rear_right"]),
        "@FIXTURE_MODE@": str(fi["mode"]),
        "@INITIAL_CARRIAGE_Y@": format(initial_carriage_y, ".17g"),
        "@SURFACE_MODE@": str(fi["surface_mode"]),
        "@TERRAIN_PHASE_X@": str(fi["terrain_phase_x"]),
        "@TERRAIN_PHASE_Z@": str(fi["terrain_phase_z"]),
        "@TERRAIN_PERIOD_X@": str(fi["terrain_period_x"]),
        "@TERRAIN_PERIOD_Z@": str(fi["terrain_period_z"]),
        "@POS_X@": str(vo["position_x_next"]),
        "@POS_Y@": str(vo["position_y_next"]),
        "@POS_Z@": str(vo["position_z_next"]),
        "@ROLL@": str(vo["roll_next"]),
        "@PITCH@": str(vo["pitch_next"]),
        "@YAW@": str(vo["yaw_next"]),
        "@VEL_X@": str(vo["velocity_x_next"]),
        "@VEL_Y@": str(vo["velocity_y_next"]),
        "@VEL_Z@": str(vo["velocity_z_next"]),
        "@ENGINE_RPM@": str(vo["engine_rpm"]),
        "@ENGINE_TORQUE@": str(vo["engine_torque"]),
        "@DRIVELINE_TORQUE@": str(vo["driveline_torque"]),
        "@WHEEL_FL@": str(vo["wheel_omega_front_left_next"]),
        "@WHEEL_FR@": str(vo["wheel_omega_front_right_next"]),
        "@WHEEL_RL@": str(vo["wheel_omega_rear_left_next"]),
        "@WHEEL_RR@": str(vo["wheel_omega_rear_right_next"]),
        "@COMP_FL@": str(vo["compression_front_left_next"]),
        "@COMP_FR@": str(vo["compression_front_right_next"]),
        "@COMP_RL@": str(vo["compression_rear_left_next"]),
        "@COMP_RR@": str(vo["compression_rear_right_next"]),
        "@SPRING_FL@": str(vo["spring_force_front_left"]),
        "@SPRING_FR@": str(vo["spring_force_front_right"]),
        "@SPRING_RL@": str(vo["spring_force_rear_left"]),
        "@SPRING_RR@": str(vo["spring_force_rear_right"]),
        "@NORMAL_FL@": str(vi.get("contact_normal_force_front_left", 0)),
        "@NORMAL_FR@": str(vi.get("contact_normal_force_front_right", 0)),
        "@NORMAL_RL@": str(vi.get("contact_normal_force_rear_left", 0)),
        "@NORMAL_RR@": str(vi.get("contact_normal_force_rear_right", 0)),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    # The viewer is emitted from this Python authority. Keep its diagnostic
    # consumer aligned with the canonical 14-value-per-wheel tire ABI, then
    # append the shared dt controller's telemetry after those 56 values.
    template = template.replace(
        "td[13*i+9];tire_energy+=td[13*i+11];tire_loss+=td[13*i+12]",
        "td[14*i+9];tire_energy+=td[14*i+11]+td[14*i+13];"
        "tire_loss+=td[14*i+12]",
    )
    dt_base = tire_output_count
    original_sim_hud = (
        f'HUD("SIM %s  DT 1/%d",simulate?"RUN":"PAUSED",{physics_hz});'
    )
    substep_hud = (
        f'HUD("SIM %s  TARGET DT 1/%d",simulate?"RUN":"PAUSED",{physics_hz});'
        f'HUD("SUBSTEP %s A %.0f OK %.0f REJ %.0f NF %.0f",'
        f'td[{dt_base}]>=.5?"LANDED":"PARTIAL",td[{dt_base + 1}],'
        f'td[{dt_base + 2}],td[{dt_base + 3}],td[{dt_base + 4}]);'
        f'HUD("SDT MIN %.3G MAX %.3G NEXT %.3G",td[{dt_base + 5}],'
        f'td[{dt_base + 6}],td[{dt_base + 8}]);'
        f'HUD("WINDOW %.6G/%.6G DISP %.3G/%.3G X%.3G",'
        f'td[{dt_base + 9}],td[{dt_base + 10}],td[{dt_base + 11}],'
        f'td[{dt_base + 12}],td[{dt_base + 13}]);'
        f'HUD("REJECT DT %.3G %.3G %.3G %.3G",td[{dt_base + 16}],'
        f'td[{dt_base + 17}],td[{dt_base + 18}],td[{dt_base + 19}]);'
    )
    template = template.replace(
        original_sim_hud,
        substep_hud,
    )
    if "@" in template:
        raise RuntimeError("native scientific viewer shell has unresolved compiler tokens")
    return template


def native_vehicle_kernel_manifest() -> Mapping[str, Any]:
    """Build the parity manifest from the same compilations used by the game."""

    vehicle_compilation = compile_symbolic_vehicle_physics()
    contact_compilation = compile_wheel_contact_ssa()
    vehicle = compile_symbolic_vehicle_physics_c()
    contact = compile_wheel_contact_c()
    fixture = compile_vehicle_roller_fixture_c()
    tire = compile_native_balloon_tire_assembly()
    if vehicle.name != "abstract_ui_vehicle_step":
        raise RuntimeError("native deployment refused a noncanonical vehicle kernel")
    if contact.name != "abstract_ui_wheel_contact":
        raise RuntimeError("native deployment refused a noncanonical contact kernel")
    if tuple(vehicle.input_names) != tuple(vehicle_compilation.function.metadata.get("argument_names", ())):
        raise RuntimeError("native vehicle ABI differs from canonical game SSA")
    if tuple(contact.input_names) != tuple(contact_compilation.function.metadata.get("argument_names", ())):
        raise RuntimeError("native contact ABI differs from canonical game SSA")
    from .vehicle_inverse_compilation import vehicle_rig_outfit_contract

    return {
        "schema": SCHEMA,
        "parity": {
            "policy": "same-compiler-input-api-abi",
            "boundary": "authored graph and parameter ABI entering compiler",
            "reduced_rig_kernel_allowed": False,
            "vehicle_ssa_sha256": _canonical_ssa_identity(vehicle_compilation),
            "contact_ssa_sha256": _canonical_ssa_identity(contact_compilation),
            "whole_graph_gate": {
                "status": "diagnostic-only",
                "compiled": ["vehicle-transition", "balloon-skin-contact", "roller-fixture"],
                "not_yet_compiled_from_shared_ssa": [
                    "mechanical-graph-constraint-position-solve",
                    "mechanical-graph-plastic-fracture-state-update",
                ],
                "launch_as_scientific-parity-rig": True,
                "note": "presentation parity is not required; renderer is an ABI consumer",
            },
        },
        "vehicle": {
            "entrypoint": vehicle.name,
            "c_sha256": _sha256_text(vehicle.source),
            "input_names": list(vehicle.input_names),
            "output_names": list(vehicle.output_names),
        },
        "contact": {
            "entrypoint": contact.name,
            "c_sha256": _sha256_text(contact.source),
            "input_names": list(contact.input_names),
            "output_names": list(contact.output_names),
        },
        "tire_appendage": {
            "entrypoint": tire.name,
            "c_sha256": _sha256_text(tire.source),
            "input_names": list(tire.input_names),
            "output_names": list(tire.output_names),
            "state_scalar_count": tire.state_scalar_count,
            "geometry": "closed deforming balloon skin against finite hard triangles",
            "integration": "passive-Kelvin-plus-implicit-bead-48-microstep-audio-rate",
            "legacy_torus_contact_linked": False,
        },
        "fixture": {
            "entrypoint": fixture.name,
            "c_sha256": _sha256_text(fixture.source),
            "input_names": list(fixture.input_names),
            "output_names": list(fixture.output_names),
            "scope": "external-test-equipment-only",
        },
        "holder_modes": {
            "cage-drive": {
                "value": HOLDER_MODES["cage-drive"],
                "excitation_target": "chassis/cage attachment forces",
                "holder": "unilateral-hard-floor",
                "separation_damping": "near-zero",
                "may_pull_wheel_down": False,
            },
            "suspension-test": {
                "value": HOLDER_MODES["suspension-test"],
                "excitation_target": "locked holder trajectory",
                "holder": "positive-position-lock",
                "may_pull_wheel_down": True,
            },
        },
        "time_integration": {
            "default_host_policy": "one fixed 120 Hz target window",
            "subdivision_authority": "Turing managed dt/superstep scheduler",
            "precision_override": "persistent multi-limb arithmetic",
            "model_api_abi_changes": False,
            "note": (
                "The display/game clock requests one window; managed scientific "
                "time chooses every accepted interior subdivision."
            ),
        },
        "audio_observation": {
            "source_nodes": [
                f"suspension.{corner}.coilover_chassis"
                for corner in ("front_left", "front_right", "rear_left", "rear_right")
            ],
            "quantity": "resultant-world-position",
            "input_pcm_is_audible": False,
        },
        "energy_diagnostics": {
            "authority": "read-only-no-force-feedback",
            "graph_operator": "weighted-incidence-laplacian-quadratic-energy",
            "per_edge": [
                "axial-strain", "elastic-energy-j", "damping-loss-w",
                "unexplained-positive-energy-residual-w",
            ],
            "appendage": [
                "balloon-membrane-strain-energy-j",
                "balloon-kelvin-dissipation-power-w", "active-contact-count",
            ],
            "presentation": "shader-energy-overlay-and-top-six-suspicious-parts",
            "smoothing_or_physics_effect": False,
        },
        "terrain_tire_batch": {
            "capacity": 8,
            "axis_order": ["terrain_batch", "wheel", "tire_vertex", "state_component"],
            "entrypoint": "vehicle_native_graph_tick_batch",
            "tire_state_entrypoint": "vehicle_native_graph_batch_tire_state",
            "reset_entrypoint": "vehicle_native_graph_batch_reset",
            "lane_state": "fully-independent-vehicle-fixture-balloon-history",
            "execution": "compiler-emitted-leading-axis-map-of-exact-scalar-graph",
            "aggregation": "presentation-only-normalized-additive-mean",
            "physics_state_is_never_averaged": True,
        },
        "arbitrary_rig_points": {
            "ownership": "external-test-equipment-runtime-state",
            "vehicle_recompile_required": False,
            "capacity": 16,
            "attachment_space": "chassis-rigid-local-coordinate",
            "wrench_boundary": "force-and-r-cross-force-into-canonical-vehicle-inputs",
            "configure_entrypoint": "vehicle_native_rig_point_configure",
            "clear_entrypoint": "vehicle_native_rig_point_clear",
            "reaction_entrypoint": "vehicle_native_rig_point_reactions",
            "configuration_values": [
                "local_xyz_m", "target_world_xyz_m", "target_velocity_xyz_m_s",
                "command_force_xyz_n", "stiffness_xyz_n_m",
                "damping_xyz_n_s_m", "maximum_force_n",
            ],
            "modes": {
                "0": "passive-six-axis-reaction-sensor",
                "1": "damped-pose-lock",
                "2": "direct-force-actuator",
                "3": "damped-velocity-servo",
            },
            "hub_note": (
                "Wheel hubs remain live articulated nodes through the compiled "
                "roller fixture ABI; they are not aliases of chassis rig points."
            ),
        },
        "rig_outfitting": vehicle_rig_outfit_contract(),
        "scientific_visualization_abi": scientific_visualization_abi(),
    }


def write_native_vehicle_kernels(
    directory: str | Path, *, outer_rate_hz: int | None = None,
    tire_microsteps: int = 48,
    assembly_profile: str = "default-car",
) -> WrittenNativeVehicleKernels:
    """Write canonical C kernels and their read-only scientific shader consumer."""

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # The game/display clock owns one target window.  The shared Turing dt
    # controller alone decides how that 1/120-second window is subdivided.
    physics_hz = int(outer_rate_hz) if outer_rate_hz is not None else 120
    profile = None
    if assembly_profile == "dually-axle":
        from .vehicle_validator_profiles import dually_validator_profile

        profile = dually_validator_profile()
    elif assembly_profile != "default-car":
        raise ValueError(f"unknown native validator assembly profile {assembly_profile!r}")
    vehicle = compile_symbolic_vehicle_physics_c()
    contact = compile_wheel_contact_c()
    fixture = compile_vehicle_roller_fixture_c()
    tire = compile_native_balloon_tire_assembly(
        tire_dimensions=(profile.tire_dimensions if profile else None),
        pneumatic_mode=(profile.tire_pneumatic_mode if profile else None),
        material_profile=(profile.tire_material_profile if profile else "configured"),
    )
    managed_tire_source, deployment_receipts = (
        _render_managed_balloon_tire_validator_source(
            tire,
            window_duration=1.0 / physics_hz,
            dt_initial=1.0 / physics_hz,
            wheel_names=(profile.wheel_names if profile else FIXTURE_CORNERS),
            tire_dimensions=(profile.tire_dimensions if profile else None),
            pneumatic_mode=(profile.tire_pneumatic_mode if profile else None),
            material_profile=(profile.tire_material_profile if profile else "configured"),
        )
    )
    # The managed program owns subdivision of one outer validator window.
    # Calling it through the retired host microstep loop would advance several
    # complete windows per vehicle tick.
    effective_tire_microsteps = 1
    balance = compile_brace_on_balance_c()
    wheel_balance = compile_wheel_mesh_balance_c()
    leveling_controller = compile_leveling_controller_c()
    leveling_sensors = compile_leveling_sensor_bank_c()
    material = compile_vehicle_member_material_c()
    from .vehicle_balloon_tire import (
        compile_balloon_bead_implicit_step_c,
        compile_balloon_contact_geometry_c,
        compile_balloon_cylinder_contact_geometry_c,
        compile_balloon_contact_impulse_c,
        compile_balloon_gas_c,
        compile_balloon_membrane_face_c,
    )
    tire_scalars = (
        compile_balloon_membrane_face_c(), compile_balloon_gas_c(),
        compile_balloon_bead_implicit_step_c(), compile_balloon_contact_geometry_c(),
        compile_balloon_cylinder_contact_geometry_c(),
        compile_balloon_contact_impulse_c(),
    )
    manifest = dict(native_vehicle_kernel_manifest())
    vehicle_path = destination / f"{vehicle.name}.c"
    contact_path = destination / f"{contact.name}.c"
    fixture_path = destination / f"{fixture.name}.c"
    tire_assembly_path = destination / "balloon_tire_appendage_step.c"
    tire_scalar_paths = tuple(destination / f"{artifact.name}.c" for artifact in tire_scalars)
    balance_path = destination / f"{balance.name}.c"
    wheel_balance_path = destination / f"{wheel_balance.name}.c"
    leveling_controller_path = destination / f"{leveling_controller.name}.c"
    material_path = destination / f"{material.name}.c"
    shell_path = destination / "vehicle_native_graph_tick.c"
    viewer_shell_path = destination / "vehicle_scientific_viewer.c"
    vehicle_path.write_text(vehicle.source, encoding="utf-8")
    contact_path.write_text(contact.source, encoding="utf-8")
    fixture_path.write_text(fixture.source, encoding="utf-8")
    tire_assembly_path.write_text(managed_tire_source, encoding="utf-8")
    for artifact, path in zip(tire_scalars, tire_scalar_paths):
        path.write_text(artifact.source, encoding="utf-8")
    balance_path.write_text(balance.source, encoding="utf-8")
    wheel_balance_path.write_text(wheel_balance.source, encoding="utf-8")
    leveling_controller_path.write_text(
        combine_c_function_artifacts(leveling_controller, leveling_sensors), encoding="utf-8")
    material_path.write_text(material.source, encoding="utf-8")
    from .vehicle_python_compilation import (
        dually_vehicle_python_compilation_inputs,
        emit_vehicle_python_graph_c,
        vehicle_python_compilation_inputs,
    )

    graph_inputs = (
        dually_vehicle_python_compilation_inputs()
        if profile is not None
        else vehicle_python_compilation_inputs()
    )
    shell_source = emit_vehicle_python_graph_c(
        inputs=graph_inputs,
    ).source
    shell_path.write_text(shell_source, encoding="utf-8")
    viewer_shell_path.write_text(
        render_native_scientific_viewer_shell(
            vehicle, contact, fixture, target_rate_hz=physics_hz,
            assembly_profile=profile,
        ),
        encoding="utf-8",
    )
    # The persistent native worker pool ships with the bundle: compiler-
    # emitted deployment spans call turing_pool_deploy_span/effect locks,
    # and the emitted C always retains its serial fallback, so linking the
    # pool is an optimization the build driver performs, never a semantic
    # obligation.
    pool_home = (
        Path(__file__).resolve().parents[1]
        / "common" / "tensors" / "accelerator_backends" / "c_backend"
    )
    for pool_name in ("turing_pool.c", "turing_pool.h"):
        (destination / pool_name).write_text(
            (pool_home / pool_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    shader_root = Path(__file__).resolve().parents[3] / "spectral-analyzer" / "csrc" / "shaders"
    vertex_shader_path = destination / "vehicle_scientific.vert.glsl"
    fragment_shader_path = destination / "vehicle_scientific.frag.glsl"
    shutil.copy2(shader_root / vertex_shader_path.name, vertex_shader_path)
    shutil.copy2(shader_root / fragment_shader_path.name, fragment_shader_path)
    manifest["shaders"] = {
        "vertex": {"path": str(shader_root / vertex_shader_path.name),
                   "sha256": _sha256_text(vertex_shader_path.read_text(encoding="utf-8"))},
        "fragment": {"path": str(shader_root / fragment_shader_path.name),
                     "sha256": _sha256_text(fragment_shader_path.read_text(encoding="utf-8"))},
        "family": "Spectral Analyzer Phong scientific diagnostic",
        "physics_authority": False,
        "encodes": ["component-type", "signed-strain", "damage-state", "joint-kind"],
    }
    manifest["native_viewer"] = {
        "shell": viewer_shell_path.name,
        "entry": "main",
        "physics_entry": "vehicle_graph_tick_vector",
        "derived_dt_hz": physics_hz,
        "presentation": "scientific-component-type-and-strain",
        "threading": {
            "physics_thread": "physics_thread_main",
            "snapshot": "generation-tagged double buffer (front/back plus "
                        "generation, DisplayDoubleBufferABI discipline)",
            "ui_commands": "queued edits applied at window boundaries",
        },
    }
    manifest["deployment"] = deployment_receipts
    manifest["deployment_pool"] = {
        "sources": ["turing_pool.c", "turing_pool.h"],
        "span_entry": "turing_pool_deploy_span",
        "effect_lock": "turing_pool_effect_lock/unlock",
        "policy": "compiler-emitted deploys carry their serial fallback; "
                  "the build driver links the pool into "
                  "vehicle_game_kernels.dll",
    }
    manifest["assembly"] = (
        assembly_manifest(load_default_car_configuration())
        if profile is None
        else {
            "identity": profile.identity,
            "profile": "dually-axle",
            "wheel_names": list(profile.wheel_names),
            "wheel_attachments": [list(row) for row in profile.wheel_attachments],
            "structural_support_positions": [
                list(row) for row in profile.structural_support_positions],
            "tire_pneumatic_mode": profile.tire_pneumatic_mode,
            "tire_material_profile": profile.tire_material_profile,
            "vehicle_selection": "loaded-validator-profile",
        }
    )
    manifest["time_integration"].update({
        "outer_rate_hz": physics_hz,
        "regular_substeps": None,
        "substep_policy": "shared-turing-dt-controller",
        "profile": "managed-dt-abstract-tensor-tire",
    })
    manifest["tire_appendage"]["integration"] = (
        "complete-managed-AbstractTensor-program-with-repository-dt-system")
    manifest["tire_appendage"]["validator_lane"] = 0
    manifest["tire_appendage"]["managed_batch_size"] = 8
    manifest["tire_appendage"]["host_requested_microsteps_ignored"] = int(
        tire_microsteps
    )
    manifest["assembly"]["balance_c_sha256"] = _sha256_text(balance.source)
    manifest["assembly"]["balance_abi"] = {
        "entrypoint": balance.name,
        "input_names": list(balance.input_names),
        "output_names": list(balance.output_names),
    }
    manifest["assembly"]["wheel_balance_c_sha256"] = _sha256_text(wheel_balance.source)
    manifest["assembly"]["wheel_balance_abi"] = {
        "entrypoint": wheel_balance.name,
        "input_names": list(wheel_balance.input_names),
        "output_names": list(wheel_balance.output_names),
    }
    manifest["assembly"]["leveling_controller_c_sha256"] = _sha256_text(
        leveling_controller.source)
    manifest["assembly"]["leveling_controller_abi"] = {
        "entrypoint": leveling_controller.name,
        "input_names": list(leveling_controller.input_names),
        "output_names": list(leveling_controller.output_names),
        "capture_policy": "fully-equipped-loadout-opposing-force-trials-before-release",
    }
    manifest["assembly"]["leveling_sensor_bank_c_sha256"] = _sha256_text(
        leveling_sensors.source)
    manifest["assembly"]["leveling_sensor_bank_abi"] = {
        "entrypoint": leveling_sensors.name,
        "input_names": list(leveling_sensors.input_names),
        "output_names": list(leveling_sensors.output_names),
        "physical_effect": "none-massless-observation-only",
    }
    _, material_graph = _render_native_material_bank(
        vehicle, material,
        mechanical_graph=(
            profile.model["mechanical_graph"] if profile else None),
    )
    manifest["mechanical_material"] = {
        **material_graph,
        "entrypoint": material.name,
        "c_sha256": _sha256_text(material.source),
        "state": "persistent-per-edge-elastic-plastic-work-hardening-fracture",
        "fracture_response": "corner-load-path-survival-opens-canonical-suspension-force-path",
        "diagnostic_entrypoint": "vehicle_native_material_diagnostics",
        "state_get_entrypoint": "vehicle_native_material_state_get",
        "state_set_entrypoint": "vehicle_native_material_state_set",
    }
    manifest["parity"]["whole_graph_gate"] = {
        "status": "compiled-material-state-active",
        "compiled": [
            "vehicle-transition", "balloon-skin-contact", "roller-fixture",
            "mechanical-graph-plastic-fracture-state-update",
        ],
        "remaining": ["full-free-node-constraint-position-solve"],
        "launch_as_scientific-parity-rig": True,
        "note": "every damage-bearing edge executes the shared material law each native tick",
    }
    manifest_path = destination / "vehicle_native.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return WrittenNativeVehicleKernels(
        vehicle_path, contact_path, fixture_path, tire_assembly_path,
        tire_scalar_paths, balance_path, wheel_balance_path, leveling_controller_path,
        material_path, shell_path,
        vertex_shader_path, fragment_shader_path, viewer_shell_path,
        manifest_path,
    )


__all__ = [
    "HOLDER_MODES",
    "SCHEMA",
    "WrittenNativeVehicleKernels",
    "split_symbolic_constants_to_double_double",
    "compile_vehicle_roller_fixture_c",
    "derive_vehicle_rig_rate_hz",
    "emit_double_double_c",
    "native_vehicle_kernel_manifest",
    "render_native_vehicle_tick_shell",
    "render_native_scientific_viewer_shell",
    "scientific_visualization_abi",
    "write_native_vehicle_kernels",
]
