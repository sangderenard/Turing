"""N-frame parity: the authored Python program against its compiled products.

For a symbolic program (a ``SymbolicEquationCompilation``) this runs the SAME
exogenous input stream through

* ``python``  -- the authored sympy equations, lambdified exactly the way the
                 Python-material validator evaluates them (the numerical
                 authority);
* ``c``       -- ``emit_ssa_function_to_c`` built by the ziglang toolchain;
* ``llvm``    -- ``emit_ssa_function_to_llvm`` built by the same toolchain;
* ``fortran`` -- ``emit_module`` built by the installed Fortran compiler,

for ``frames`` consecutive frames, feeding each product's OWN outputs back
into its next frame (``x_next`` -> ``x``, ``x_next`` -> ``x_previous``,
``x_state`` -> ``x``), so drift accumulates per backend exactly as it would in
a running simulation.  ``--open-loop`` instead feeds every backend the Python
trajectory, isolating single-frame error from accumulated drift.

Nothing here rebuilds a vehicle: every program below compiles in seconds.
The vehicle body is available under ``--programs vehicle_body`` but is not a
default because its C emission is large.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.symbolic_equation_compiler import SymbolicEquationCompilation  # noqa: E402


# --------------------------------------------------------------------------
# programs
# --------------------------------------------------------------------------

CORNERS = ("front_left", "front_right", "rear_left", "rear_right")

Driver = Callable[[int, dict[str, float]], None]


def _roller_fixture() -> tuple[SymbolicEquationCompilation, dict[str, float], Driver]:
    from src.compiler.vehicle_native_deployment import compile_vehicle_roller_fixture_ssa

    compilation = compile_vehicle_roller_fixture_ssa()
    hub_y, carriage_y = 0.31, 0.19
    feeds = {
        "dt": 1.0 / 1024.0, "gravity": -9.81, "floor_y": -0.75,
        "carriage_mass": 12.0, "neutral_buoyancy": 1.0, "passive_damping": 8.0,
        "lock_stiffness": 24_000.0, "lock_damping": 1_200.0,
        "maximum_actuator_force": 18_000.0, "mode": 0.0, "surface_mode": 0.0,
        "terrain_period_x": 4.0, "terrain_period_z": 4.0,
        "terrain_velocity_x": 0.35, "terrain_velocity_z": -0.1,
    }
    for corner in CORNERS:
        feeds[f"carriage_y_{corner}"] = carriage_y
        feeds[f"command_y_{corner}"] = carriage_y
        feeds[f"hub_y_{corner}"] = hub_y
        feeds[f"mode_{corner}"] = 1.0

    def drive(frame: int, inputs: dict[str, float]) -> None:
        phase = 2.0 * math.pi * frame / 96.0
        for index, corner in enumerate(CORNERS):
            inputs[f"command_y_{corner}"] = carriage_y + 0.04 * math.sin(phase + index)
            inputs[f"command_velocity_y_{corner}"] = (
                0.04 * math.cos(phase + index) * 2.0 * math.pi / 96.0 * 1024.0)
            inputs[f"hub_velocity_y_{corner}"] = 0.02 * math.sin(phase * 0.5)
            inputs[f"roller_reaction_{corner}"] = 40.0 * max(0.0, math.sin(phase * 0.25))

    return compilation, feeds, drive


def _member_material() -> tuple[SymbolicEquationCompilation, dict[str, float], Driver]:
    from src.compiler.vehicle_mechanical_material import compile_vehicle_member_material_ssa

    compilation = compile_vehicle_member_material_ssa()
    feeds = {
        "dt": 1.0 / 1024.0,
        "youngs_modulus_pa": 2.0e11, "shear_modulus_pa": 8.0e10,
        "initial_yield_stress_pa": 3.5e8, "ultimate_stress_pa": 4.5e8,
        "hardening_modulus_pa": 2.0e9, "fracture_plastic_strain": 0.2,
        "hardening_fragility": 0.1, "material_volume_m3": 1.0e-5,
        "axial_viscosity_pa_s": 1.0e5, "bending_viscosity_pa_s": 1.0e5,
        "shear_viscosity_pa_s": 1.0e5,
    }

    def drive(frame: int, inputs: dict[str, float]) -> None:
        # Past yield (3.5e8 / 2e11 = 1.75e-3) so the plastic branch is live.
        omega = 2.0 * math.pi / 50.0
        inputs["axial_strain"] = 4.0e-3 * math.sin(omega * frame)
        inputs["axial_strain_rate"] = 4.0e-3 * omega * math.cos(omega * frame) * 1024.0
        inputs["bending_strain"] = 1.0e-3 * math.sin(omega * frame * 0.5)
        inputs["bending_strain_rate"] = 1.0e-3 * omega * 0.5 * math.cos(omega * frame * 0.5) * 1024.0
        inputs["shear_strain"] = 5.0e-4 * math.cos(omega * frame)
        inputs["shear_strain_rate"] = -5.0e-4 * omega * math.sin(omega * frame) * 1024.0

    return compilation, feeds, drive


def _wheel_contact() -> tuple[SymbolicEquationCompilation, dict[str, float], Driver]:
    from src.compiler.abstract_ui_vehicles import compile_wheel_contact_ssa

    compilation = compile_wheel_contact_ssa()
    feeds = {
        "dt": 1.0 / 120.0, "support": 1.0, "suspension_travel": 0.34,
        "maximum_compression_speed": 1.25, "tire_pressure": 155_000.0,
        "minimum_contact_area": 0.006, "maximum_contact_area": 0.045,
        "corner_weight": 1_520.0, "slip_transition_speed": 0.38,
    }

    def drive(frame: int, inputs: dict[str, float]) -> None:
        phase = 2.0 * math.pi * frame / 80.0
        for name in inputs:
            if name.startswith("compression") and not name.endswith("_previous"):
                inputs[name] = 0.05 + 0.03 * math.sin(phase)
            if "slip" in name and "transition" not in name:
                inputs[name] = 0.2 * math.sin(phase * 1.7)
            if "velocity" in name and "speed" not in name:
                inputs[name] = 3.0 + 2.0 * math.cos(phase)

    return compilation, feeds, drive


def _vehicle_body() -> tuple[SymbolicEquationCompilation, dict[str, float], Driver]:
    from src.compiler.abstract_ui_vehicles import (
        compile_symbolic_vehicle_physics, load_default_car_configuration,
    )

    compilation = compile_symbolic_vehicle_physics()
    feeds = {name: 0.0 for name in compilation.function.metadata["argument_names"]}
    feeds.update(load_default_car_configuration().parameter_defaults())
    feeds.update({"dt": 1.0 / 1024.0, "position_y": 0.9, "yaw_cos": 1.0,
                  "gravity": -9.81})

    def drive(frame: int, inputs: dict[str, float]) -> None:
        inputs["throttle"] = 0.3 if frame > 8 else 0.0
        for corner in CORNERS:
            inputs[f"contact_normal_force_{corner}"] = 3_700.0

    return compilation, feeds, drive


def _tire_gas():
    from src.compiler.vehicle_balloon_tire import compile_balloon_gas_ssa

    compilation = compile_balloon_gas_ssa()
    feeds = {"current_volume_m3": 0.05, "gas_polytropic_exponent": 1.4,
             "minimum_volume_fraction": 0.2, "reference_pressure_pa": 135_000.0,
             "reference_temperature_k": 293.0, "reference_volume_m3": 0.05}

    def drive(frame: int, inputs: dict[str, float]) -> None:
        inputs["current_volume_m3"] = 0.05 * (1.0 + 0.1 * math.sin(2.0 * math.pi * frame / 40.0))

    return compilation, feeds, drive


def _tire_bead_implicit_step():
    from src.compiler.vehicle_balloon_tire import compile_balloon_bead_implicit_step_ssa

    compilation = compile_balloon_bead_implicit_step_ssa()
    feeds = {"bead_damping_n_s_per_m": 9_200.0, "bead_stiffness_n_per_m": 2.4e6,
             "dt": 2.44140625e-4, "vertex_mass_kg": 0.0972222,
             "free_velocity_x": 0.1, "free_velocity_y": 0.0, "free_velocity_z": 0.0,
             "rim_center_x": 0.0, "rim_center_y": 0.0, "rim_center_z": 0.0,
             "target_x": 0.30, "target_y": 0.0, "target_z": 0.0,
             "target_velocity_x": 0.0, "target_velocity_y": 0.0, "target_velocity_z": 0.0,
             "vertex_x": 0.31, "vertex_y": 0.0, "vertex_z": 0.0}

    def drive(frame: int, inputs: dict[str, float]) -> None:
        inputs["target_y"] = 0.002 * math.sin(2.0 * math.pi * frame / 30.0)

    # A bead vertex integrates in place: its next position/velocity are its
    # own next inputs (names the generic ``_next -> base`` rule cannot see).
    feedback = {f"position_{axis}_next": f"vertex_{axis}" for axis in "xyz"}
    feedback.update({f"velocity_{axis}_next": f"free_velocity_{axis}" for axis in "xyz"})
    return compilation, feeds, drive, feedback


def _tire_contact_impulse():
    from src.compiler.vehicle_balloon_tire import compile_balloon_contact_impulse_ssa

    compilation = compile_balloon_contact_impulse_ssa()
    feeds = {"contact_active": 1.0, "friction_coefficient": 0.9,
             "inverse_effective_mass_per_kg": 1.0 / 0.0972222,
             "normal_x": 0.0, "normal_y": 1.0, "normal_z": 0.0, "restitution": 0.1,
             "velocity_x": 0.5, "velocity_y": -0.3, "velocity_z": 0.1}

    def drive(frame: int, inputs: dict[str, float]) -> None:
        phase = 2.0 * math.pi * frame / 50.0
        inputs["velocity_x"] = 0.5 * math.cos(phase)
        inputs["velocity_y"] = -0.3 - 0.2 * math.sin(phase)
        inputs["velocity_z"] = 0.1 * math.sin(2.0 * phase)

    return compilation, feeds, drive


def _tire_membrane_face():
    from src.compiler.vehicle_balloon_tire import compile_balloon_membrane_face_ssa

    compilation = compile_balloon_membrane_face_ssa()
    # One rest triangle of the authored tire's scale (edges ~0.19 m), with its
    # metric, inverse and area derived consistently from those rest points.
    r0 = np.array([0.0, 0.0, 0.0]); r1 = np.array([0.19, 0.0, 0.0]); r2 = np.array([0.08, 0.17, 0.0])
    e1, e2 = r1 - r0, r2 - r0
    metric = np.array([[e1 @ e1, e1 @ e2], [e2 @ e1, e2 @ e2]])
    inverse = np.linalg.inv(metric)
    lam, mu = 6.2e6, 4.1e6
    feeds = {
        "gas_pressure_pa": 135_000.0, "reference_pressure_pa": 135_000.0,
        "lame_lambda_pa": lam, "lame_mu_pa": mu,
        "membrane_damping_lambda_pa_s": 5_400.0, "membrane_damping_mu_pa_s": 3_600.0,
        "natural_metric_00": metric[0, 0], "natural_metric_01": metric[0, 1], "natural_metric_11": metric[1, 1],
        "rest_inverse_00": inverse[0, 0], "rest_inverse_01": inverse[0, 1],
        "rest_inverse_10": inverse[1, 0], "rest_inverse_11": inverse[1, 1],
        "rest_area_m2": 0.5 * float(np.linalg.norm(np.cross(e1, e2))),
        "orthotropic_q11_pa": lam + 2 * mu, "orthotropic_q22_pa": lam + 2 * mu,
        "orthotropic_q12_pa": lam, "orthotropic_q16_pa": 0.0, "orthotropic_q26_pa": 0.0,
        "orthotropic_q66_pa": mu, "skin_thickness_m": 0.012,
    }
    for index, point in enumerate((r0, r1, r2)):
        for axis, value in zip("xyz", point):
            feeds[f"r{index}_{axis}"] = float(value)
            feeds[f"x{index}_{axis}"] = float(value)
            feeds[f"v{index}_{axis}"] = 0.0

    def drive(frame: int, inputs: dict[str, float]) -> None:
        phase = 2.0 * math.pi * frame / 60.0
        # Stretch vertex 1 along x and lift vertex 2 out of plane, slowly.
        inputs["x1_x"] = 0.19 * (1.0 + 0.01 * math.sin(phase))
        inputs["x2_z"] = 0.004 * math.sin(0.5 * phase)
        inputs["v1_x"] = 0.19 * 0.01 * math.cos(phase) * 2.0 * math.pi / 60.0 * 4096.0
        inputs["v2_z"] = 0.004 * 0.5 * math.cos(0.5 * phase) * 2.0 * math.pi / 60.0 * 4096.0

    return compilation, feeds, drive


PROGRAMS: dict[str, Callable[[], tuple]] = {
    "roller_fixture": _roller_fixture,
    "member_material": _member_material,
    "wheel_contact": _wheel_contact,
    "vehicle_body": _vehicle_body,
    "tire_gas": _tire_gas,
    "tire_bead_implicit_step": _tire_bead_implicit_step,
    "tire_contact_impulse": _tire_contact_impulse,
    "tire_membrane_face": _tire_membrane_face,
}


# --------------------------------------------------------------------------
# feedback: which outputs become which inputs on the next frame
# --------------------------------------------------------------------------

def feedback_map(input_names: Sequence[str], output_names: Sequence[str]) -> dict[str, str]:
    """output name -> input name, by the repository's naming conventions."""

    inputs = set(input_names)
    mapping: dict[str, str] = {}
    for output in output_names:
        for suffix in ("_next", "_state"):
            if output.endswith(suffix):
                base = output[: -len(suffix)]
                for candidate in (base, f"{base}_previous"):
                    if candidate in inputs:
                        mapping[output] = candidate
                        break
                break
    return mapping


# --------------------------------------------------------------------------
# backends: each returns a callable inputs-vector -> outputs-vector
# --------------------------------------------------------------------------

def _runtime_maximum(*values: Any) -> Any:
    result = values[0]
    for value in values[1:]:
        result = max(result, value)
    return result


def _runtime_minimum(*values: Any) -> Any:
    result = values[0]
    for value in values[1:]:
        result = min(result, value)
    return result


def python_backend(compilation: SymbolicEquationCompilation) -> Callable[[np.ndarray], np.ndarray]:
    """The authority: sympy lambdify with the validator's runtime modules."""

    import sympy

    metadata = compilation.function.metadata
    arguments = tuple(metadata["argument_names"])
    outputs = tuple(metadata["output_names"])
    by_name = {str(equation.lhs): equation.rhs for equation in compilation.equations}
    modules = [{
        "Max": _runtime_maximum, "Min": _runtime_minimum,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "Abs": abs,
        "tanh": math.tanh, "exp": math.exp,
    }, "math"]
    function = sympy.lambdify(
        tuple(sympy.Symbol(name) for name in arguments),
        tuple(by_name[name] for name in outputs),
        modules=modules, cse=True,
    )

    def evaluate(vector: np.ndarray) -> np.ndarray:
        return np.asarray(function(*(float(v) for v in vector)), dtype=np.float64)

    return evaluate


def c_backend(compilation: SymbolicEquationCompilation, directory: Path,
              optimization: str) -> Callable[[np.ndarray], np.ndarray]:
    from src.compiler.ssa_c_backend import emit_ssa_function_to_c

    name = compilation.function.name
    artifact = emit_ssa_function_to_c(compilation.module, name, entry_name=name)
    if not artifact.complete:
        raise RuntimeError("C shortfalls: " + "; ".join(s.reason for s in artifact.shortfalls[:5]))
    assert tuple(artifact.input_names) == tuple(compilation.function.metadata["argument_names"])
    assert tuple(artifact.output_names) == tuple(compilation.function.metadata["output_names"])
    artifact.compile(directory / "c", optimization=optimization)
    entry = artifact.entry()
    out_count = len(artifact.output_names)

    def evaluate(vector: np.ndarray) -> np.ndarray:
        inputs = np.ascontiguousarray(vector, dtype=np.float64)
        outputs = np.zeros(out_count, dtype=np.float64)
        entry(inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
              outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return outputs

    return evaluate


def llvm_backend(compilation: SymbolicEquationCompilation, directory: Path,
                 optimization: str) -> Callable[[np.ndarray], np.ndarray]:
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )

    name = compilation.function.name
    artifact = emit_ssa_function_to_llvm(compilation.module, name)
    if not artifact.complete:
        raise RuntimeError("LLVM shortfalls: " + "; ".join(s.reason for s in artifact.shortfalls[:5]))
    compile_artifact(artifact, directory=directory / "llvm", optimization=optimization)
    arguments = tuple(compilation.function.metadata["argument_names"])
    outputs = tuple(compilation.function.metadata["output_names"])
    input_ids = [int(compilation.input_ids[n]) for n in arguments]
    output_ids = [int(compilation.output_ids[n]) for n in outputs]
    execution = prepare_artifact_execution(
        artifact, {value_id: np.float64(0.0) for value_id in input_ids})

    def evaluate(vector: np.ndarray) -> np.ndarray:
        for value_id, value in zip(input_ids, vector):
            execution.buffers[value_id][...] = float(value)
        execution.run()
        return np.asarray([float(execution.buffers[value_id]) for value_id in output_ids])

    return evaluate


def fortran_backend(compilation: SymbolicEquationCompilation, directory: Path,
                    ) -> Callable[[np.ndarray], np.ndarray] | None:
    from src.compiler.ssa_fortran_backend import compile_module, emit_module, fortran_compiler

    compiler = fortran_compiler()
    if compiler is None:
        return None
    name = compilation.function.name
    function = compilation.function
    ret = next(i for b in function.blocks.values() for i in b.instrs if i.op == "Ret")
    module = emit_module(
        compilation.module, name=f"{name}_parity", outputs={name: tuple(ret.args)},
        progress=lambda _message: None,
    )
    if not module.complete:
        raise RuntimeError("Fortran shortfalls: " + "; ".join(s.reason for s in module.shortfalls[:5]))
    library_path = compile_module(module, directory=directory / "fortran")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(Path(compiler).parent))
    library = ctypes.CDLL(str(library_path))
    api = module.api.entry_point(name)
    native = getattr(library, api.symbol)
    ctypes_by_dtype = {"float64": ctypes.c_double, "real64": ctypes.c_double,
                       "double": ctypes.c_double, "float32": ctypes.c_float,
                       "int32": ctypes.c_int32, "int64": ctypes.c_int64,
                       "bool": ctypes.c_bool}
    arguments = tuple(function.metadata["argument_names"])
    outputs = tuple(function.metadata["output_names"])
    input_ids = {int(compilation.input_ids[n]): index for index, n in enumerate(arguments)}
    # One SSA value can be returned under several names (CSE folds identical
    # outputs, e.g. a per-corner force that is the same expression for every
    # corner), so an output value id fills EVERY slot that names it.
    output_slots: dict[int, list[int]] = {}
    for index, n in enumerate(outputs):
        output_slots.setdefault(int(compilation.output_ids[n]), []).append(index)
    plan: list[tuple[str, Any, Any, Any]] = []   # (role, ctype, slot(s), extent)
    for parameter in api.parameters:
        ctype = ctypes_by_dtype.get(str(parameter.dtype).casefold(), ctypes.c_double)
        if parameter.role == "extent":
            plan.append(("extent", ctype, None, int(parameter.name.rsplit("_", 1)[-1])))
            continue
        value_id = int(parameter.name[1:])
        if parameter.role in {"input", "inout"}:
            plan.append((f"in:{parameter.passing}", ctype, input_ids[value_id], None))
        else:
            plan.append(("out", ctype, tuple(output_slots[value_id]), None))
    native.restype = None

    def evaluate(vector: np.ndarray) -> np.ndarray:
        result = np.zeros(len(outputs), dtype=np.float64)
        keep: list[Any] = []
        call: list[Any] = []
        types: list[Any] = []
        for role, ctype, slot, extent in plan:
            if role == "extent":
                call.append(ctype(extent)); types.append(ctype)
            elif role == "in:value":
                call.append(ctype(float(vector[slot]))); types.append(ctype)
            elif role.startswith("in:"):
                cell = ctype(float(vector[slot])); keep.append(cell)
                call.append(ctypes.byref(cell)); types.append(ctypes.POINTER(ctype))
            else:
                cell = ctype(0.0); keep.append((slot, cell))
                call.append(ctypes.byref(cell)); types.append(ctypes.POINTER(ctype))
        native.argtypes = types
        native(*call)
        for item in keep:
            if isinstance(item, tuple):
                for index in item[0]:
                    result[index] = float(item[1].value)
        return result

    return evaluate


# --------------------------------------------------------------------------
# the parity run
# --------------------------------------------------------------------------

def run_parity(
    program: str, *, frames: int, backends: Sequence[str], optimization: str = "O2",
    open_loop: bool = False, directory: Path | None = None,
) -> dict[str, Any]:
    program_parts = PROGRAMS[program]()
    compilation, feeds, drive = program_parts[:3]
    explicit_feedback = dict(program_parts[3]) if len(program_parts) > 3 else {}
    metadata = compilation.function.metadata
    arguments = tuple(metadata["argument_names"])
    outputs = tuple(metadata["output_names"])
    feedback = {**feedback_map(arguments, outputs), **explicit_feedback}
    workdir = Path(directory or tempfile.mkdtemp(prefix=f"frame_parity_{program}_"))

    evaluators: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    unavailable: dict[str, str] = {}
    for backend in backends:
        try:
            if backend == "python":
                evaluators[backend] = python_backend(compilation)
            elif backend == "c":
                evaluators[backend] = c_backend(compilation, workdir, optimization)
            elif backend == "llvm":
                evaluators[backend] = llvm_backend(compilation, workdir, optimization)
            elif backend == "fortran":
                evaluator = fortran_backend(compilation, workdir)
                if evaluator is None:
                    unavailable[backend] = "no Fortran compiler installed"
                else:
                    evaluators[backend] = evaluator
            else:
                raise ValueError(f"unknown backend {backend!r}")
        except RuntimeError as error:
            # A backend that cannot emit or build this program is a finding
            # (reported), not a reason to lose the other backends' numbers.
            if backend == "python":
                raise
            unavailable[backend] = str(error)
    if "python" not in evaluators:
        raise ValueError("the python authority must be among the backends")

    index_of = {name: index for index, name in enumerate(arguments)}
    base = np.zeros(len(arguments), dtype=np.float64)
    for name, value in feeds.items():
        if name in index_of:
            base[index_of[name]] = float(value)
    state = {backend: base.copy() for backend in evaluators}
    trajectories = {backend: np.zeros((frames, len(outputs))) for backend in evaluators}
    seconds = {backend: 0.0 for backend in evaluators}

    import time as _time

    for frame in range(frames):
        exogenous: dict[str, float] = {}
        drive(frame, exogenous)
        for backend, evaluator in evaluators.items():
            vector = state[backend]
            for name, value in exogenous.items():
                if name in index_of:
                    vector[index_of[name]] = float(value)
            started = _time.perf_counter()
            trajectories[backend][frame] = evaluator(vector)
            seconds[backend] += _time.perf_counter() - started
        for backend in evaluators:
            source = "python" if open_loop else backend
            produced = trajectories[source][frame]
            for output, target in feedback.items():
                state[backend][index_of[target]] = produced[outputs.index(output)]

    reference = trajectories["python"]

    # Conditioning baseline: the authority against ITSELF with one ULP of
    # rounding residue injected into every fed-back state after frame 0.
    # A backend whose closed-loop divergence stays within a few multiples
    # of this is inside the program's own sensitivity to rounding; only a
    # divergence far beyond it can be a backend defect.
    perturbed = np.zeros_like(reference)
    vector = base.copy()
    for frame in range(frames):
        exogenous = {}
        drive(frame, exogenous)
        for name, value in exogenous.items():
            if name in index_of:
                vector[index_of[name]] = float(value)
        perturbed[frame] = evaluators["python"](vector)
        for output, target in feedback.items():
            value = perturbed[frame][outputs.index(output)]
            if frame == 0:
                value = np.nextafter(value, np.inf) if value != 0.0 else 1.0e-19
            vector[index_of[target]] = value
    self_finite = np.isfinite(perturbed) & np.isfinite(reference)
    self_difference = np.abs(perturbed - reference)
    self_column_scale = np.max(np.abs(reference), axis=0, keepdims=True)
    self_scale = np.maximum(np.maximum(np.abs(reference), np.abs(perturbed)),
                            1.0e-12 * np.maximum(self_column_scale, 1.0e-300))
    self_relative = np.where(self_finite, self_difference / self_scale, np.inf)
    one_ulp_sensitivity = {
        "max_abs_divergence": float(self_difference[self_finite].max()) if self_finite.any() else float("inf"),
        "max_rel_divergence": float(self_relative[self_finite].max()) if self_finite.any() else float("inf"),
    }

    report: dict[str, Any] = {
        "program": program, "function": compilation.function.name,
        "frames": frames, "mode": "open-loop" if open_loop else "closed-loop",
        "optimization": optimization, "symbolic_cache_hit": bool(compilation.cache_hit),
        "inputs": len(arguments), "outputs": len(outputs),
        "feedback": feedback, "unavailable": unavailable, "backends": {},
        "python_one_ulp_sensitivity": one_ulp_sensitivity,
        # Mean wall time of one evaluation per backend, microseconds.  The
        # Python figure is the sympy-lambdified authority (interpreter cost,
        # not a kernel); the native figures are the kernels themselves.
        "microseconds_per_call": {
            backend: 1.0e6 * seconds[backend] / max(frames, 1) for backend in evaluators
        },
    }
    for backend, values in trajectories.items():
        if backend == "python":
            continue
        finite = np.isfinite(values) & np.isfinite(reference)
        difference = np.abs(values - reference)
        # Relative to the larger of the two values, floored per output at
        # 1e-12 of that output's own trajectory scale, so a reference that
        # passes through zero does not report an infinite relative error.
        column_scale = np.max(np.abs(reference), axis=0, keepdims=True)
        scale = np.maximum(np.maximum(np.abs(reference), np.abs(values)),
                           1.0e-12 * np.maximum(column_scale, 1.0e-300))
        relative = np.where(finite, difference / scale, np.inf)
        worst = np.unravel_index(int(np.argmax(np.where(finite, difference, np.inf))), difference.shape)
        per_frame_max = np.max(np.where(finite, difference, np.inf), axis=1)
        max_rel = float(relative[finite].max()) if finite.any() else float("inf")
        report["backends"][backend] = {
            "max_abs_error": float(difference[finite].max()) if finite.any() else float("inf"),
            "max_rel_error": max_rel,
            "rel_error_over_one_ulp_sensitivity": (
                max_rel / one_ulp_sensitivity["max_rel_divergence"]
                if one_ulp_sensitivity["max_rel_divergence"] > 0.0 else float("inf")),
            "worst_frame": int(worst[0]), "worst_output": outputs[int(worst[1])],
            "first_frame_over_1e-9_abs": next(
                (int(f) for f in range(frames) if per_frame_max[f] > 1.0e-9), None),
            "abs_error_by_frame_quartiles": [
                float(np.percentile(per_frame_max, q)) for q in (0, 25, 50, 75, 100)],
            "non_finite": int((~np.isfinite(values)).sum()),
        }
    names = [b for b in trajectories if b != "python"]
    report["backend_vs_backend_max_abs"] = {
        f"{a}/{b}": float(np.max(np.abs(trajectories[a] - trajectories[b])))
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", nargs="+", default=("roller_fixture", "member_material"),
                        choices=tuple(PROGRAMS))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--backends", nargs="+", default=("python", "c", "llvm", "fortran"))
    parser.add_argument("--optimization", default="O2")
    parser.add_argument("--open-loop", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    args = parser.parse_args()
    reports = []
    for program in args.programs:
        report = run_parity(program, frames=args.frames, backends=tuple(args.backends),
                            optimization=args.optimization, open_loop=args.open_loop)
        reports.append(report)
        sensitivity = report["python_one_ulp_sensitivity"]
        print(f"== {program} ({report['function']}) {report['mode']} frames={report['frames']} "
              f"cache_hit={report['symbolic_cache_hit']} feedback={len(report['feedback'])} "
              f"outputs | python 1-ULP self-divergence: abs={sensitivity['max_abs_divergence']:.3e} "
              f"rel={sensitivity['max_rel_divergence']:.3e}")
        for backend, row in report["backends"].items():
            print(f"  {backend:8s} max_abs={row['max_abs_error']:.3e} max_rel={row['max_rel_error']:.3e} "
                  f"(x{row['rel_error_over_one_ulp_sensitivity']:.2g} of 1-ULP sensitivity) "
                  f"worst=frame {row['worst_frame']} {row['worst_output']} "
                  f"first>1e-9 at frame {row['first_frame_over_1e-9_abs']} non_finite={row['non_finite']}")
        for pair, value in report["backend_vs_backend_max_abs"].items():
            print(f"  {pair:14s} max_abs={value:.3e}")
        print("  time/call: " + "  ".join(
            f"{backend}={value:.1f}us" for backend, value in report["microseconds_per_call"].items()))
        for backend, reason in report["unavailable"].items():
            print(f"  {backend}: unavailable ({reason})")
    if args.json is not None:
        args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
