"""The Python validator's coupled simulation, with presentation outside the ABI.

The graph and its laws come from vehicle_python_compilation. This module owns
the same validator input packing, next-state feedback and DT window for eager
and native execution. It does not replace the coupled rig with a tire fixture.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics
from src.common.tensors import AbstractTensor

from .vehicle_python_compilation import (
    VehiclePythonSSALowering, _managed_native_feeds_by_id,
    dually_vehicle_python_compilation_inputs, vehicle_python_runtime_bindings,
)
from .vehicle_balloon_tire_stability import tire_microstep_count


# These are the persistent publications of _PythonVehicleMaterial.tick plus
# the harness vectors that it packs afresh before each candidate step.
STATE_FIELDS = (
    "vehicle_in", "contact_in", "fixture_in", "vehicle_out",
    "vehicle_input", "contact_input", "fixture_global", "fixture_wheel",
    "fixture_surface", "tire_input", "tire_state", "tire_output",
    "tire_previous_hub", "tire_previous_basis", "tire_previous_angle",
    "tire_previous_plane", "material_state", "roller_anchor",
    "roller_anchor_valid", "tire_initialized", "tire_history_valid",
    "wheel_assembly_alpha", "compression", "compression_velocity",
    "wheel_angle", "wheel_omega", "outer_dt",
    "fixture_output", "surface_output", "rig_reactions",
    "material_diagnostics", "pillar_reactions",
    "vehicle_output", "contact_output",
)
RETURN_FIELDS = (*STATE_FIELDS, "last_displacement", "telemetry", "advanced", "dt_next", "hard_failure")


class ValidatorSimulationState:
    """Physical record; no viewer, lock, callback, or Python material object."""

    def __init__(self, **values):
        self.__dict__.update(values)

    def dt_limit_hint(self):
        return float(self.declared_dt_s)

    def copy_shallow(self):
        return (
            self.vehicle_in.copy(), self.contact_in.copy(), self.fixture_in.copy(),
            self.vehicle_out.copy(), self.vehicle_input.copy(), self.contact_input.copy(),
            self.fixture_global.copy(), self.fixture_wheel.copy(), self.fixture_surface.copy(),
            self.tire_input.copy(), self.tire_state.copy(), self.tire_output.copy(),
            self.tire_previous_hub.copy(), self.tire_previous_basis.copy(),
            self.tire_previous_angle.copy(), self.tire_previous_plane.copy(),
            self.material_state.copy(), self.roller_anchor.copy(),
            self.roller_anchor_valid.copy(), self.tire_initialized.copy(),
            self.tire_history_valid.copy(), self.wheel_assembly_alpha.copy(),
            self.compression.copy(), self.compression_velocity.copy(),
            self.wheel_angle.copy(), self.wheel_omega.copy(), self.outer_dt.copy(),
            self.fixture_output.copy(), self.surface_output.copy(),
            self.rig_reactions.copy(), self.material_diagnostics.copy(),
            self.pillar_reactions.copy(),
            self.vehicle_output.copy(), self.contact_output.copy(),
        )

    def restore(self, saved):
        self.vehicle_in[...] = saved[0]
        self.contact_in[...] = saved[1]
        self.fixture_in[...] = saved[2]
        self.vehicle_out[...] = saved[3]
        self.vehicle_input[...] = saved[4]
        self.contact_input[...] = saved[5]
        self.fixture_global[...] = saved[6]
        self.fixture_wheel[...] = saved[7]
        self.fixture_surface[...] = saved[8]
        self.tire_input[...] = saved[9]
        self.tire_state[...] = saved[10]
        self.tire_output[...] = saved[11]
        self.tire_previous_hub[...] = saved[12]
        self.tire_previous_basis[...] = saved[13]
        self.tire_previous_angle[...] = saved[14]
        self.tire_previous_plane[...] = saved[15]
        self.material_state[...] = saved[16]
        self.roller_anchor[...] = saved[17]
        self.roller_anchor_valid[...] = saved[18]
        self.tire_initialized[...] = saved[19]
        self.tire_history_valid[...] = saved[20]
        self.wheel_assembly_alpha[...] = saved[21]
        self.compression[...] = saved[22]
        self.compression_velocity[...] = saved[23]
        self.wheel_angle[...] = saved[24]
        self.wheel_omega[...] = saved[25]
        self.outer_dt[...] = saved[26]
        self.fixture_output[...] = saved[27]
        self.surface_output[...] = saved[28]
        self.rig_reactions[...] = saved[29]
        self.material_diagnostics[...] = saved[30]
        self.pillar_reactions[...] = saved[31]
        self.vehicle_output[...] = saved[32]
        self.contact_output[...] = saved[33]


def validator_layout():
    from .abstract_ui_vehicles import compile_symbolic_vehicle_physics, compile_wheel_contact_ssa
    from .vehicle_native_deployment import compile_vehicle_roller_fixture_ssa

    vehicle = compile_symbolic_vehicle_physics().function.metadata
    return {
        "vehicle": tuple(vehicle["argument_names"]),
        "output": tuple(vehicle["output_names"]),
        "contact": tuple(compile_wheel_contact_ssa().function.metadata["argument_names"]),
        "fixture": tuple(compile_vehicle_roller_fixture_ssa().function.metadata["argument_names"]),
    }


def simulation_source(prepared, layout):
    """Specialize only named vector layout; the coupled graph is unchanged."""
    vi, ci, fi = ({name: i for i, name in enumerate(layout[key])}
                  for key in ("vehicle", "contact", "fixture"))
    from .vehicle_native_deployment import FIXTURE_CORNERS

    lines = [
        "def validator_simulation_advance(material, dt):",
        "    material.telemetry[0] = material.telemetry[0] + 1.0",
        "    material.telemetry[3] = dt",
        "    previous = material.tire_state[0, :, :, 0:3] + 0.0",
        "    previous_velocity = material.tire_state[0, :, :, 3:6] + 0.0",
        f"    material.vehicle_in[{vi['dt']}] = dt",
        f"    material.fixture_in[{fi['dt']}] = dt",
        "    material.vehicle_input[:, :] = material.vehicle_in",
        "    material.contact_input[:, :, :] = 0.0",
    ]
    for wheel, corner in enumerate(FIXTURE_CORNERS):
        for axis, name in enumerate(("attachment_x", "attachment_y", "attachment_z")):
            lines.append(f"    material.contact_input[:, {wheel}, {3 + axis}] = material.contact_in[{wheel * len(ci) + ci[name]}]")
        for column, stem in enumerate(("hub_y", "hub_velocity_y", "carriage_y", "carriage_velocity_y", "command_y", "command_velocity_y", "roller_reaction", "mode")):
            index = fi.get(f"{stem}_{corner}")
            rhs = "0.0" if index is None else f"material.fixture_in[{index}]"
            lines.append(f"    material.fixture_wheel[:, {wheel}, {column}] = {rhs}")
        lines.append(f"    material.wheel_assembly_alpha[:, {wheel}] = material.vehicle_in[{vi[f'assembly_alpha_{corner}']}]")
        for name in ("compression", "compression_velocity", "wheel_angle", "wheel_omega"):
            lines.append(f"    material.{name}[:, {wheel}] = material.vehicle_in[{vi[f'{name}_{corner}']}]")
    for column, name in enumerate(("dt", "mode", "gravity", "floor_y", "carriage_mass", "neutral_buoyancy", "passive_damping", "lock_stiffness", "lock_damping", "maximum_actuator_force")):
        lines.append(f"    material.fixture_global[:, {column}] = material.fixture_in[{fi[name]}]")
    for column, name in enumerate(("surface_mode", "terrain_phase_x", "terrain_phase_z", "terrain_velocity_x", "terrain_velocity_z", "terrain_period_x", "terrain_period_z")):
        lines.append(f"    material.fixture_surface[:, {column}] = material.fixture_in[{fi[name]}]")
    lines.extend((
        "    material.outer_dt[:] = dt",
        "    material.microstep_count = tire_microstep_count(dt, material.tire_critical_dt_s, material.tire_dt_fraction)",
        "    result = vehicle_graph_tick_vector(" + ", ".join(f"material.{name}" for name in prepared.feeds) + ")",
        "    material.vehicle_out[:] = result[0][0, :]",
    ))
    for wheel, corner in enumerate(FIXTURE_CORNERS):
        lines.extend((
            f"    material.contact_in[{wheel * len(ci) + ci['support']}] = result[1][0, {wheel}, 6]",
            f"    material.fixture_in[{fi[f'carriage_y_{corner}']}] = result[2][0, {wheel}, 0]",
            f"    material.fixture_in[{fi[f'carriage_velocity_y_{corner}']}] = result[2][0, {wheel}, 1]",
        ))
    publications = {
        "tire_input": "result[4]", "tire_state": "result[5]", "tire_output": "result[6]",
        "tire_previous_hub": "result[7][0]", "tire_previous_basis": "result[7][1]",
        "tire_previous_angle": "result[7][2]", "tire_previous_plane": "result[7][3]",
        "material_state": "result[9]", "roller_anchor": "result[11]",
        "roller_anchor_valid": "result[13]", "tire_initialized": "result[14]",
        "tire_history_valid": "result[15]", "fixture_output": "result[2]",
        "surface_output": "result[3]", "rig_reactions": "result[8]",
        "material_diagnostics": "result[10]", "pillar_reactions": "result[12]",
        "vehicle_output": "result[0]", "contact_output": "result[1]",
    }
    lines.extend(f"    material.{name} = {value}" for name, value in publications.items())
    for index, name in enumerate(layout["output"]):
        if name.endswith("_next") and name[:-5] in vi:
            lines.append(f"    material.vehicle_in[{vi[name[:-5]]}] = material.vehicle_out[{index}]")
    lines.extend(ADVANCE_METRICS.splitlines())
    lines.extend((
        "", "def validator_simulation_window(material, targets, controller, window_duration, dt_initial):",
        "    material.telemetry[:] = 0.0",
        "    advanced, dt_next, metrics = run_superstep(",
        "        material, window_duration, dt_initial, 0.03, targets, controller,",
        "        validator_simulation_advance, allow_increase_mid_round=True,",
        "        max_retries=None, rollback_threshold_multiplier=material.rollback_threshold_multiplier,",
        "        rollback=material.rollback_enabled)",
        "    hard_failure = bool(metrics.hard_failure)",
        "    return (" + ", ".join(
            f"material.{name}" if name not in {"advanced", "dt_next", "hard_failure"} else name
            for name in RETURN_FIELDS) + ")",
    ))
    return prepared.source + "\n\n" + "\n".join(lines) + "\n"


ADVANCE_METRICS = '''    position = material.tire_state[0, :, :, 0:3]
    velocity = material.tire_state[0, :, :, 3:6]
    delta = position - previous
    material.last_displacement = (delta * delta).sum(dim=-1).sqrt()
    displacement = material.last_displacement.max()
    maximum_velocity = (velocity * velocity).sum(dim=-1).sqrt().max()
    pressure = material.tire_output[:, :, 6]
    vertex_mass = material.tire_input[0, 2]
    kinetic_after = 0.5 * vertex_mass * (velocity * velocity).sum()
    kinetic_before = 0.5 * vertex_mass * (previous_velocity * previous_velocity).sum()
    stored_energy = kinetic_after + (material.tire_output[0, :, 11].sum() + material.tire_output[0, :, 13].sum())
    exchange_power = abs(kinetic_after - kinetic_before) / max(dt, 1.0e-30) + abs(material.tire_output[0, :, 12].sum())
    finite = material.tire_state[0].isfinite().all() and pressure.isfinite().all()
    physical = bool(finite and position.abs().max() < 100.0 and pressure.max() < material.rated_pressure_pa * 3.0 and pressure.min() >= 0.0)
    accepted = physical and displacement <= 0.006 * material.rollback_threshold_multiplier
    if accepted:
        material.telemetry[1] = material.telemetry[1] + 1.0
    else:
        material.telemetry[2] = material.telemetry[2] + 1.0
    material.telemetry[4] = displacement
    material.telemetry[5] = maximum_velocity
    material.telemetry[6] = max(material.telemetry[6], displacement)
    material.telemetry[7] = stored_energy
    material.telemetry[8] = exchange_power
    channel_values = AbstractTensor.tensor([stored_energy, exchange_power, 0.0, 0.0, 0.0, 0.0, 0.0, displacement, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    channel_present = AbstractTensor.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau_present = exchange_power > 0.0
    return physical, Metrics(max_vel=maximum_velocity, max_flux=maximum_velocity,
        div_inf=0.0, mass_err=0.0, error_channels=channel_values,
        error_present=channel_present, advanced_dt=dt,
        pub_tau=AbstractTensor.tensor([stored_energy / exchange_power if tau_present else 0.0]),
        pub_tau_present=AbstractTensor.tensor([float(tau_present)]),
        pub_contract=AbstractTensor.tensor([1.0 if tau_present else 0.0]),
        pub_dt_limit=AbstractTensor.tensor([0.0]),
        pub_dt_limit_present=AbstractTensor.tensor([0.0]),
        pub_values=channel_values, pub_present=channel_present,
        pub_limits=AbstractTensor.zeros_like(channel_values),
        pub_limits_present=AbstractTensor.zeros_like(channel_present))
'''


def simulation_inputs(batch_size=8):
    from .vehicle_validator_profiles import dually_validator_profile

    profile = dually_validator_profile()
    prepared = dually_vehicle_python_compilation_inputs(
        batch_size, rig_point_count=len(profile.structural_support_positions))
    layout = validator_layout()
    values = {name: value.copy() if isinstance(value, np.ndarray) else value
              for name, value in prepared.feeds.items()}
    values.update(
        vehicle_in=np.zeros(len(layout["vehicle"])),
        contact_in=np.zeros(4 * len(layout["contact"])),
        fixture_in=np.zeros(len(layout["fixture"])),
        vehicle_out=np.zeros(len(layout["output"])),
        vehicle_output=np.zeros((batch_size, len(layout["output"]))),
        contact_output=np.zeros((batch_size, 4, 9)),
        last_displacement=np.zeros(values["tire_state"].shape[1:3]),
        fixture_output=np.zeros((batch_size, 4, 5)),
        surface_output=np.zeros((batch_size, 5)),
        rig_reactions=np.zeros((batch_size, len(profile.structural_support_positions), 6)),
        material_diagnostics=np.zeros((*values["material_state"].shape[:2], 8)),
        pillar_reactions=np.zeros((batch_size, 4)), telemetry=np.zeros(9),
        declared_dt_s=float(values["tire_input"][0, 0]),
        tire_critical_dt_s=1.0, tire_dt_fraction=0.3,
        rollback_enabled=False, rollback_threshold_multiplier=2.0,
        rated_pressure_pa=float(profile.rated_pressure_pa),
    )
    return SimpleNamespace(
        prepared=prepared, layout=layout,
        source=simulation_source(prepared, layout),
        material=ValidatorSimulationState(**values),
        controller=STController(dt_min=None, dt_max=1.0 / 1024.0),
        targets=Targets(cfl=0.22, div_max=1.0, mass_max=1.0,
                        error_limits=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                        error_limits_present=AbstractTensor.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                        energy_exchange_fraction=0.1),
    )


def eager_functions(inputs):
    namespace = {"AbstractTensor": AbstractTensor, "Metrics": Metrics,
                 "run_superstep": run_superstep, "tire_microstep_count": tire_microstep_count,
                 **vehicle_python_runtime_bindings(include_configured_vehicle=False)}
    exec(inputs.source, namespace)
    return namespace["validator_simulation_advance"], namespace["validator_simulation_window"]


def load_validator_state(inputs, live_material, vehicle_in, contact_in, fixture_in,
                         vehicle_out, *, tire_dt_fraction=0.3, rollback=False,
                         rollback_threshold_multiplier=2.0):
    """Copy a complete outer-tick input; immutable topology keeps its dtype."""
    state = inputs.material
    with AbstractTensor.use_backend("numpy"):
        for name in inputs.prepared.feeds:
            value = live_material.feeds[name]
            if hasattr(value, "data"):
                value = np.array(live_material._data(value), copy=True)
                value = AbstractTensor.tensor(value)
            setattr(state, name, value)
        for name, value in (
            ("vehicle_in", vehicle_in), ("contact_in", contact_in),
            ("fixture_in", fixture_in), ("vehicle_out", vehicle_out),
        ):
            setattr(state, name, AbstractTensor.tensor(np.asarray(tuple(value))))
        for name in (*STATE_FIELDS, "last_displacement", "telemetry"):
            value = getattr(state, name)
            if isinstance(value, np.ndarray):
                setattr(state, name, AbstractTensor.tensor(value.copy()))
    state.declared_dt_s = live_material.declared_tire_dt_s
    state.tire_critical_dt_s = live_material.tire_critical_dt_s
    state.tire_dt_fraction = float(tire_dt_fraction)
    state.rollback_enabled = bool(rollback)
    state.rollback_threshold_multiplier = float(rollback_threshold_multiplier)
    return state


def simulation_contract(inputs):
    from .extraction_contract import ExtractionContract

    home = Path(__file__).resolve().parents[2] / "extraction_contracts"
    policy = ExtractionContract(home / "program_extraction.yaml")
    base = policy.program_abi.receipt()
    records = {name: base["records"][name] for name in ("Targets", "STController", "Metrics")}
    fields = {}
    for name, value in vars(inputs.material).items():
        if isinstance(value, np.ndarray):
            fields[name] = dict(storage="span", dtype=str(value.dtype), rank=value.ndim,
                                shape=list(value.shape), mutable=name in STATE_FIELDS or name in {"telemetry", "last_displacement"})
        else:
            fields[name] = dict(storage="scalar", dtype="bool" if isinstance(value, bool) else "float64",
                                mutable=name == "microstep_count")
    records["ValidatorSimulationState"] = {
        "identity": __name__ + ".ValidatorSimulationState", "fields": fields}
    bindings = [binding for binding in base["bindings"] if binding["record"] in records]
    bindings += [
        {"function": "*", "parameter": "material", "record": "ValidatorSimulationState"},
        {"function": "step_with_dt_control_used", "parameter": "state", "record": "ValidatorSimulationState"},
    ]
    values = [dict(function="validator_simulation_window", parameter=name, storage="scalar",
                   dtype="float64", rank=0, python_type="builtins.float")
              for name in ("window_duration", "dt_initial")]
    return policy.with_program_abi(dict(records=records, bindings=bindings, values=values)).with_execution_file(
        home / "vehicle_full_native_execution.yaml")


def lower_simulation(inputs, progress=None, diagnostic_directory=None):
    from .fortran_c_shell import lower_ast_source_to_ssa
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference

    def save_graph(graph):
        if diagnostic_directory is not None:
            from .project_compilation_product import _dump_resolved_process_graph
            with (Path(diagnostic_directory) / "resolved-process-graph.pkl").open("wb") as stream:
                _dump_resolved_process_graph(graph, stream)

    module, outputs, exports = lower_ast_source_to_ssa(
        inputs.source, "validator_simulation_window", name="validator_simulation",
        python_bindings={"AbstractTensor": AbstractTensor, "Metrics": Metrics,
                         "run_superstep": run_superstep, "tire_microstep_count": tire_microstep_count},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        linked_process_graphs=inputs.prepared.linked_process_graphs,
        extraction_contract=simulation_contract(inputs), retain=(ValidatorSimulationState,),
        runtime_closure_only=True, progress=progress,
        resolved_process_graph_sink=save_graph,
    )
    roots = [name for name in module.functions if name.endswith("__validator_simulation_window")]
    if len(roots) != 1:
        raise RuntimeError(f"expected one simulation entry, got {roots!r}")
    return VehiclePythonSSALowering(module, roots[0], dict(outputs), tuple(exports))


def build_simulation(directory, batch_size=8, progress=print):
    from .ssa_c_backend import emit_ssa_to_c
    from .ssa_self_check import run_all

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    inputs = simulation_inputs(batch_size)
    (destination / "simulation-source.py").write_text(inputs.source, encoding="utf-8")
    progress("Preparing the coupled validator graph and explicit simulation ABI")
    lowered = lower_simulation(inputs, progress=progress, diagnostic_directory=destination)
    with (destination / "repository-ssa.pkl").open("wb") as stream:
        pickle.dump((lowered.module, lowered.outputs, lowered.exports), stream)
    findings = run_all(lowered.module)
    if findings:
        raise RuntimeError(f"simulation SSA has structural findings: {findings[:10]}")
    artifact = emit_ssa_to_c(lowered.module, lowered.root_name, entry_name="validator_simulation_native")
    if not artifact.complete:
        raise RuntimeError(f"simulation C emission incomplete: {artifact.shortfalls[:10]}")
    root = lowered.module.functions[lowered.root_name]
    return_rows = [tuple(int(value.id) for value in instruction.args)
                   for block in root.blocks.values() for instruction in block.instrs
                   if instruction.op == "Ret"]
    if len(return_rows) != 1 or len(return_rows[0]) != len(RETURN_FIELDS):
        raise RuntimeError(f"simulation return contract does not match its {len(RETURN_FIELDS)} publications: {return_rows}")
    progress("Compiling complete simulation C at O0")
    artifact.compile(destination, optimization="O0")
    with (destination / "artifact.pkl").open("wb") as stream:
        pickle.dump(artifact, stream)
    manifest = dict(schema="turing.validator-simulation-native.v1", batch_size=batch_size,
                    entrypoint=artifact.name, root_name=lowered.root_name,
                    library=artifact.library_path.name, layout=inputs.layout,
                    return_ids=dict(zip(RETURN_FIELDS, return_rows[0])),
                    python_callbacks=False, controller="src.common.dt_system.dt_controller.run_superstep",
                    optimization="O0", source_sha256=hashlib.sha256(inputs.source.encode()).hexdigest())
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class NativeValidatorSimulation:
    """One native DT window, called by the Python viewer's existing worker."""

    def __init__(self, directory, live_material):
        root = Path(directory).resolve()
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "turing.validator-simulation-native.v1":
            raise ValueError("not a coupled validator simulation artifact")
        if self.manifest["batch_size"] != live_material.lanes:
            raise ValueError("native simulation lane count differs from the live validator")
        self.inputs = simulation_inputs(live_material.lanes)
        if hashlib.sha256(self.inputs.source.encode()).hexdigest() != self.manifest["source_sha256"]:
            raise ValueError("native simulation source changed; rebuild the artifact")
        # These are compiler artifacts from the user's local build directory.
        with (root / "repository-ssa.pkl").open("rb") as stream:
            module, outputs, exports = pickle.load(stream)
        self.lowered = VehiclePythonSSALowering(module, self.manifest["root_name"], outputs, exports)
        with (root / "artifact.pkl").open("rb") as stream:
            self.artifact = pickle.load(stream)
        self.artifact.library_path = root / self.manifest["library"]
        self.artifact._entry = None
        self.live_material = live_material

    def step(self, window, dt_initial, controller, targets, vehicle_in, contact_in,
             fixture_in, vehicle_out, **options):
        state = load_validator_state(self.inputs, self.live_material, vehicle_in,
                                     contact_in, fixture_in, vehicle_out, **options)
        feeds = _managed_native_feeds_by_id(self.lowered, dict(
            material=state, controller=controller, targets=targets,
            window_duration=float(window), dt_initial=float(dt_initial)))
        feeds = {key: self.live_material._data(value) if isinstance(value, AbstractTensor)
                 else value for key, value in feeds.items()}
        returned = self.manifest["return_ids"]
        dtypes = dict(zip(self.artifact.buffer_order, self.artifact.buffer_dtypes))
        from .ssa_c_backend import _numpy_dtype
        for name, value_id in returned.items():
            if value_id not in feeds:
                shape = (() if name in {"advanced", "dt_next", "hard_failure"}
                         else np.asarray(self.live_material._data(getattr(state, name))).shape)
                feeds[value_id] = np.zeros(shape or (1,), dtype=_numpy_dtype(dtypes[value_id]))
        missing = set(self.artifact.buffer_order) - set(feeds)
        if missing:
            raise RuntimeError(f"native simulation has unnamed public inputs: {sorted(missing)}")
        execution = self.artifact.prepare_execution(feeds)
        execution.run()
        values = {name: execution.buffers[value_id].copy() for name, value_id in returned.items()}
        advanced = float(values["advanced"].reshape(-1)[0])
        hard_failure = bool(values["hard_failure"].reshape(-1)[0])
        if not np.isfinite(advanced) or advanced <= 0.0 or hard_failure:
            raise RuntimeError(f"native controller failed to advance: {advanced=}, {hard_failure=}")
        # Update the real Python controller from its mutable ABI, so later
        # windows do not silently start from the construction defaults.
        function = self.lowered.module.functions[self.lowered.root_name]
        payloads, presence = {}, {}
        for argument in function.args:
            accounting = argument.accounting or {}
            name = accounting.get("program_abi_field")
            if accounting.get("program_abi_parameter") != "controller" or not name or "." in name:
                continue
            if accounting.get("linked_call_frame_storage"):
                continue
            value = execution.buffers[int(argument.id)].reshape(-1)[0].item()
            if accounting.get("program_abi_optional_presence"):
                presence[name] = bool(value) == bool(accounting.get("program_abi_optional_present_when", True))
            else:
                if name in payloads and payloads[name] != value:
                    raise RuntimeError(f"ambiguous native controller publication for {name}")
                payloads[name] = value
        for name, value in payloads.items():
            old = getattr(controller, name)
            setattr(controller, name, None if presence.get(name) is False else
                    type(old)(value) if old is not None else float(value))
        with AbstractTensor.use_backend("numpy"):
            for name in STATE_FIELDS:
                shape = np.asarray(self.live_material._data(getattr(state, name))).shape
                value = values[name].reshape(shape)
                dtype = np.asarray(self.live_material._data(getattr(state, name))).dtype
                setattr(state, name, AbstractTensor.tensor(value.astype(dtype, copy=False)))
            for name in self.inputs.prepared.feeds:
                if name in STATE_FIELDS:
                    self.live_material.feeds[name] = getattr(state, name)
        for name, destination in (("vehicle_in", vehicle_in), ("contact_in", contact_in),
                                  ("fixture_in", fixture_in), ("vehicle_out", vehicle_out)):
            for index, value in enumerate(values[name].reshape(-1)):
                destination[index] = float(value)
        result = (
            state.vehicle_output, state.contact_output, state.fixture_output,
            state.surface_output, state.tire_input, state.tire_state, state.tire_output,
            (state.tire_previous_hub, state.tire_previous_basis, state.tire_previous_angle, state.tire_previous_plane),
            state.rig_reactions, state.material_state, state.material_diagnostics,
            state.roller_anchor, state.pillar_reactions, state.roller_anchor_valid,
            state.tire_initialized, state.tire_history_valid,
        )
        self.live_material.accept_tick_result(result, contact_in, fixture_in, vehicle_out)
        return SimpleNamespace(advanced=advanced,
                               dt_next=float(values["dt_next"].reshape(-1)[0]),
                               telemetry=values["telemetry"].reshape(-1),
                               displacement=values["last_displacement"].reshape(
                                   np.asarray(self.live_material._data(state.last_displacement)).shape))
