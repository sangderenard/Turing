"""Probe the compiled balloon appendage without the chassis or roller fixture.

This is deliberately an ABI-level diagnostic: it calls the same exported C
appendage that the native vehicle shell calls.  A radial scale below one gives
the gas/membrane system a clean, symmetric volume perturbation; no contact law
or fixture can then obscure whether the tire restores or creates energy.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.abstract_ui_vehicles import load_default_car_configuration
from src.compiler.vehicle_balloon_tire import balloon_tire_graph_abi
from src.compiler.vehicle_balloon_tire_native import compile_native_balloon_tire_assembly
from src.compiler.vehicle_balloon_tire_diagnostics import balloon_tire_state_diagnostics


def _signed_volume(state, faces) -> float:
    value = 0.0
    for a, b, c in faces:
        ax, ay, az = state[6 * a:6 * a + 3]
        bx, by, bz = state[6 * b:6 * b + 3]
        cx, cy, cz = state[6 * c:6 * c + 3]
        value += (
            ax * (by * cz - bz * cy)
            + ay * (bz * cx - bx * cz)
            + az * (bx * cy - by * cx)
        ) / 6.0
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--rate-hz", type=float, default=16_384.0)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--radial-scale", type=float, default=1.0)
    parser.add_argument("--start-charge", type=float, default=1.0)
    parser.add_argument("--end-charge", type=float, default=1.0)
    parser.add_argument("--inflate-seconds", type=float, default=0.0)
    parser.add_argument("--spin-rad-s", type=float, default=0.0)
    parser.add_argument("--bending-stiffness-nm", type=float)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    assembly = compile_native_balloon_tire_assembly()
    topology = balloon_tire_graph_abi(load_default_car_configuration().source)["topology"]
    input_count = len(assembly.input_names)
    output_count = len(assembly.output_names)
    inputs = (ctypes.c_double * input_count)()
    state = (ctypes.c_double * assembly.state_scalar_count)()
    outputs = (ctypes.c_double * output_count)()
    library = ctypes.CDLL(str(args.library.resolve()))
    library.balloon_tire_appendage_defaults.argtypes = [ctypes.POINTER(ctypes.c_double)]
    library.balloon_tire_appendage_initialize.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    library.balloon_tire_appendage_step.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.balloon_tire_appendage_defaults(inputs)
    by_input = {name: index for index, name in enumerate(assembly.input_names)}
    by_output = {name: index for index, name in enumerate(assembly.output_names)}
    inputs[by_input["dt"]] = 1.0 / args.rate_hz
    inputs[by_input["gravity_y"]] = 0.0
    if args.bending_stiffness_nm is not None:
        inputs[by_input["bending_stiffness_nm"]] = args.bending_stiffness_nm
    inputs[by_input["gas_charge_fraction"]] = args.start_charge
    for corner in ("front_left", "front_right", "rear_left", "rear_right"):
        inputs[by_input[f"{corner}.hub_angular_velocity_z"]] = args.spin_rad_s
    library.balloon_tire_appendage_initialize(inputs, state)
    rest_state = tuple(state)
    for index in range(0, assembly.state_scalar_count, 6):
        state[index] *= args.radial_scale
        state[index + 1] *= args.radial_scale
        state[index + 2] *= args.radial_scale

    peak_force = 0.0
    peak_moment = 0.0
    peak_strain = 0.0
    peak_kinetic = 0.0
    peak_laplacian = 0.0
    finite = True
    samples = max(1, round(args.rate_hz * args.seconds))
    events = []
    sample_interval = max(1, samples // 20)
    for step in range(samples):
        elapsed = step / args.rate_hz
        ramp = 1.0 if args.inflate_seconds <= 0.0 else min(1.0, elapsed / args.inflate_seconds)
        charge = args.start_charge + ramp * (args.end_charge - args.start_charge)
        inputs[by_input["gas_charge_fraction"]] = charge
        for corner in ("front_left", "front_right", "rear_left", "rear_right"):
            inputs[by_input[f"{corner}.hub_angle_rad"]] = args.spin_rad_s * elapsed
        library.balloon_tire_appendage_step(inputs, state, outputs)
        finite = finite and all(math.isfinite(value) for value in state)
        force = math.sqrt(sum(outputs[by_output[f"front_left.rim_force_{axis}_n"]] ** 2 for axis in "xyz"))
        moment = math.sqrt(sum(outputs[by_output[f"front_left.rim_moment_{axis}_nm"]] ** 2 for axis in "xyz"))
        peak_force = max(peak_force, force)
        peak_moment = max(peak_moment, moment)
        peak_strain = max(peak_strain, outputs[by_output["front_left.strain_energy_j"]])
        if not finite:
            break

        if step % sample_interval == 0 or step + 1 == samples:
            state_view = tuple(state[:assembly.state_scalar_count // 4])
            diagnostic = balloon_tire_state_diagnostics(
                state_view, rest_state[:assembly.state_scalar_count // 4],
                topology,
                inputs[by_input["vertex_mass_kg"]],
            )
            peak_kinetic = max(peak_kinetic, diagnostic["kinetic_energy_j"])
            peak_laplacian = max(peak_laplacian, diagnostic["laplace_beltrami_energy"])
            events.append({
                "time_s": elapsed,
                "charge_fraction": charge,
                "pressure_pa": outputs[by_output["front_left.gas_pressure_pa"]],
                "volume_ratio": outputs[by_output["front_left.volume_ratio"]],
                "strain_energy_j": outputs[by_output["front_left.strain_energy_j"]],
                "bending_energy_j": outputs[by_output["front_left.bending_energy_j"]],
                "damping_removal_power_w": -outputs[by_output["front_left.dissipation_power_w"]],
                **diagnostic,
            })

    final_volume_ratio = outputs[by_output["front_left.volume_ratio"]]
    final_diag = events[-1] if events else {}
    if not finite:
        verdict = "non-finite integration failure"
    elif final_volume_ratio < 0.7:
        verdict = "global membrane collapse; gas/restoring work did not arrest volume loss"
    elif abs(final_volume_ratio - 1.0) < 0.01 and final_diag.get("kinetic_energy_j", 1.0) < 1.0:
        verdict = "stable return to the reference breathing state"
    elif final_diag.get("mode_class") == "axisymmetric-section-mode":
        verdict = "bounded axisymmetric section ringing; damping has not reached quiescence"
    elif final_diag.get("mode_class") == "localized-high-frequency-mode":
        verdict = "localized high-frequency membrane mode dominates"
    else:
        verdict = "bounded but not quiescent; inspect energy timeline and top vertices"

    result = {
        "finite": finite,
        "rate_hz": args.rate_hz,
        "seconds_requested": args.seconds,
        "steps_completed": step + 1,
        "radial_scale": args.radial_scale,
        "spin_rad_s": args.spin_rad_s,
        "bending_stiffness_nm": inputs[by_input["bending_stiffness_nm"]],
        "pressure_pa": outputs[by_output["front_left.gas_pressure_pa"]],
        "volume_ratio": outputs[by_output["front_left.volume_ratio"]],
        "minimum_skin_y_m": outputs[by_output["front_left.minimum_skin_y_m"]],
        "strain_energy_j": outputs[by_output["front_left.strain_energy_j"]],
        "bending_energy_j": outputs[by_output["front_left.bending_energy_j"]],
        "peak_strain_energy_j": peak_strain,
        "peak_kinetic_energy_j": peak_kinetic,
        "peak_laplace_beltrami_energy": peak_laplacian,
        "peak_rim_force_n": peak_force,
        "peak_rim_moment_nm": peak_moment,
        "diagnostic_verdict": verdict,
        "final_diagnostics": final_diag,
        "events": [] if args.summary_only else events,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
