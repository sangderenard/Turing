"""Run the compiled native vehicle graph without rendering and audit stability."""

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
from src.compiler.vehicle_native_deployment import derive_vehicle_rig_rate_hz


CORNERS = ("front_left", "front_right", "rear_left", "rear_right")


def _values(names: list[str], values: dict[str, float]):
    array_type = ctypes.c_double * len(names)
    return array_type(*(float(values.get(name, 0.0)) for name in names))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--engine-start-seconds", type=float, default=10.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    config = load_default_car_configuration()
    source = config.source
    hz = derive_vehicle_rig_rate_hz(config)
    dt = 1.0 / hz

    vehicle_names = list(manifest["vehicle"]["input_names"])
    output_names = list(manifest["vehicle"]["output_names"])
    contact_names = list(manifest["contact"]["input_names"])
    fixture_names = list(manifest["fixture"]["input_names"])
    tire_names = list(manifest["tire_appendage"]["output_names"])
    vehicle_defaults = {name: 0.0 for name in vehicle_names}
    vehicle_defaults.update(config.parameter_defaults())
    vehicle_defaults.update({
        "dt": dt, "position_y": 0.9, "yaw_cos": 1.0,
        "engine_enabled": 0.0, "throttle": 0.0,
        "forward_gear_ratio": 1.0, "transfer_case_ratio": 1.0,
        "drive_fraction_front_left": .21, "drive_fraction_front_right": .21,
        "drive_fraction_rear_left": .29, "drive_fraction_rear_right": .29,
    })
    tire = source["tires"]
    section = float(tire["toroid_section_radius_m"])
    major = float(tire["radius"]) - section
    one_contact = {name: 0.0 for name in contact_names}
    one_contact.update({
        "support": 1.0, "normal_y": 1.0, "forward_x": 1.0, "right_z": 1.0,
        "tire_pressure": float(tire["pressure_pa"]),
        "tire_major_radius": major, "tire_section_radius": section,
    })
    attachments = ((1.2, -.52, -.78), (1.2, -.52, .78),
                   (-1.2, -.52, -.78), (-1.2, -.52, .78))
    contact_values = []
    for attachment in attachments:
        current = dict(one_contact)
        current.update(dict(zip(("attachment_x", "attachment_y", "attachment_z"), attachment)))
        contact_values.extend(float(current.get(name, 0.0)) for name in contact_names)

    initial_hub_y = .38
    roller_distance = float(tire["radius"]) + .13
    initial_carriage_y = initial_hub_y - math.sqrt(max(0.0, roller_distance ** 2 - .18 ** 2))
    fixture_defaults = {name: 0.0 for name in fixture_names}
    fixture_defaults.update({
        "dt": dt, "gravity": -9.81, "floor_y": -.75,
        "carriage_mass": 12.0, "neutral_buoyancy": 1.0,
        "passive_damping": 8.0, "lock_stiffness": 24_000.0,
        "lock_damping": 1_200.0, "maximum_actuator_force": 18_000.0,
        "mode": 0.0, "surface_mode": 0.0,
        "terrain_period_x": 4.0, "terrain_period_z": 4.0,
    })
    for corner in CORNERS:
        fixture_defaults[f"carriage_y_{corner}"] = initial_carriage_y
        fixture_defaults[f"command_y_{corner}"] = initial_carriage_y
        fixture_defaults[f"hub_y_{corner}"] = initial_hub_y

    library = ctypes.CDLL(str(bundle / "vehicle_game_kernels.dll"))
    tick = library.vehicle_native_graph_tick
    tick.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4
    diagnostic = library.vehicle_native_tire_diagnostics
    diagnostic.argtypes = [ctypes.POINTER(ctypes.c_double)]
    energy_diagnostic = library.vehicle_native_energy_diagnostics
    energy_diagnostic.argtypes = [ctypes.POINTER(ctypes.c_double)]
    library.vehicle_native_reset()
    configure_rig_point = library.vehicle_native_rig_point_configure
    configure_rig_point.argtypes = [ctypes.c_int, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_double)]
    clear_rig_point = library.vehicle_native_rig_point_clear
    clear_rig_point.argtypes = [ctypes.c_int]
    rig_reactions = library.vehicle_native_rig_point_reactions
    rig_reactions.argtypes = [ctypes.POINTER(ctypes.c_double)]
    vehicle_in = _values(vehicle_names, vehicle_defaults)
    contact_in = (ctypes.c_double * len(contact_values))(*contact_values)
    fixture_in = _values(fixture_names, fixture_defaults)
    vehicle_out = (ctypes.c_double * len(output_names))()
    tire_out = (ctypes.c_double * len(tire_names))()
    energy_out = (ctypes.c_double * 4)()
    reaction_out = (ctypes.c_double * (16 * 6))()
    vi = {name: index for index, name in enumerate(vehicle_names)}
    vo = {name: index for index, name in enumerate(output_names)}
    feedback = tuple((vo[name], vi[name[:-5]]) for name in output_names
                     if name.endswith("_next") and name[:-5] in vi)

    # The qualification stand is external laboratory equipment.  These five
    # runtime fixtures attach at the four chassis-pan corners and centre, hold
    # the initially quiescent pose, measure the complete reaction wrench, and
    # then release progressively.  Nothing here changes or recompiles the
    # vehicle equations or their JSON parameters.
    chassis = source["chassis"]
    pan_y = -0.5 * float(chassis["height"])
    pan_x = 0.92 * float(chassis["half_length"])
    pan_z = 0.92 * float(chassis["half_width"])
    pan_points = ((pan_x, pan_y, pan_z), (pan_x, pan_y, -pan_z),
                  (-pan_x, pan_y, pan_z), (-pan_x, pan_y, -pan_z),
                  (0.0, pan_y, 0.0))
    fixture_release_times = (2.0, 2.5, 3.0, 3.5, 1.0)
    rig_value_type = ctypes.c_double * 19
    for slot, local in enumerate(pan_points):
        target = (local[0], vehicle_defaults["position_y"] + local[1], local[2])
        values = rig_value_type(*(
            *local, *target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            80_000.0, 120_000.0, 80_000.0,
            4_000.0, 6_000.0, 4_000.0, 60_000.0,
        ))
        configure_rig_point(slot, 1, values)

    failures: list[str] = []
    maximum_abs_state = 0.0
    maximum_tire_kinetic = 0.0
    maximum_tire_energy = 0.0
    maximum_tire_loss = 0.0
    minimum_contacts_after_settle = math.inf
    maximum_anchor_reaction_n = 0.0
    released: set[int] = set()
    report_each = max(1, round(hz))
    steps = round(args.seconds * hz)
    for step in range(steps):
        time_s = step * dt
        for slot, release_time in enumerate(fixture_release_times):
            if slot not in released and time_s >= release_time:
                clear_rig_point(slot)
                released.add(slot)
        vehicle_in[vi["engine_enabled"]] = float(time_s >= args.engine_start_seconds)
        tick(vehicle_in, contact_in, fixture_in, vehicle_out)
        for output_index, input_index in feedback:
            vehicle_in[input_index] = vehicle_out[output_index]
        diagnostic(tire_out)
        energy_diagnostic(energy_out)
        rig_reactions(reaction_out)
        values = tuple(vehicle_out)
        if not all(math.isfinite(value) for value in values):
            failures.append(f"non-finite vehicle state at {time_s:.6f} s")
            break
        maximum_abs_state = max(maximum_abs_state, max(map(abs, values), default=0.0))
        tire_kinetic = energy_out[0]
        tire_energy = sum(tire_out[13 * corner + 11] for corner in range(4))
        tire_loss = sum(tire_out[13 * corner + 12] for corner in range(4))
        contacts = sum(tire_out[13 * corner + 9] for corner in range(4))
        if not all(math.isfinite(value) for value in (tire_kinetic, tire_energy, tire_loss, contacts)):
            failures.append(f"non-finite tire diagnostic at {time_s:.6f} s")
            break
        maximum_tire_energy = max(maximum_tire_energy, tire_energy)
        maximum_tire_kinetic = max(maximum_tire_kinetic, tire_kinetic)
        maximum_tire_loss = max(maximum_tire_loss, tire_loss)
        for slot in range(len(pan_points)):
            force_offset = 6 * slot
            reaction = math.sqrt(sum(reaction_out[force_offset + axis] ** 2
                                     for axis in range(3)))
            maximum_anchor_reaction_n = max(maximum_anchor_reaction_n, reaction)
        if time_s >= 1.0:
            minimum_contacts_after_settle = min(minimum_contacts_after_settle, contacts)
        if abs(vehicle_out[vo["position_y_next"]]) > 5.0:
            failures.append(f"chassis escaped fixture at {time_s:.6f} s")
            break
        if tire_kinetic > 1.0e8 or tire_energy > 1.0e8 or tire_loss > 1.0e10:
            failures.append(f"balloon energy runaway at {time_s:.6f} s")
            break
        if step % report_each == 0:
            print(json.dumps({
                "simulated_seconds": round(time_s, 4),
                "position_y_m": vehicle_out[vo["position_y_next"]],
                "velocity_y_m_s": vehicle_out[vo["velocity_y_next"]],
                "pitch_rad": vehicle_out[vo["pitch_next"]],
                "contacts": contacts,
                "tire_kinetic_energy_j": tire_kinetic,
                "tire_strain_energy_j": tire_energy,
                "tire_dissipation_power_w": tire_loss,
                "maximum_anchor_reaction_n": maximum_anchor_reaction_n,
                "active_pan_anchors": len(pan_points) - len(released),
            }), flush=True)
    if minimum_contacts_after_settle == math.inf:
        minimum_contacts_after_settle = 0.0
    if not failures and minimum_contacts_after_settle <= 0:
        failures.append("all balloon/terrain contacts dropped after the settle window")
    result = {
        "schema": "turing.native-vehicle-soak.v1",
        "pass": not failures,
        "failures": failures,
        "simulated_seconds_requested": args.seconds,
        "physics_hz": hz,
        "engine_start_seconds": args.engine_start_seconds,
        "maximum_abs_published_state": maximum_abs_state,
        "maximum_tire_kinetic_energy_j": maximum_tire_kinetic,
        "maximum_tire_strain_energy_j": maximum_tire_energy,
        "maximum_tire_dissipation_power_w": maximum_tire_loss,
        "maximum_pan_anchor_reaction_n": maximum_anchor_reaction_n,
        "pan_anchor_release_times_s": fixture_release_times,
        "minimum_contact_vertices_after_one_second": minimum_contacts_after_settle,
        "final": {name: vehicle_out[index] for index, name in enumerate(output_names)},
    }
    destination = (args.report or bundle / "native_soak_report.json").resolve()
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(destination), **{k: result[k] for k in (
        "pass", "failures", "maximum_tire_strain_energy_j",
        "minimum_contact_vertices_after_one_second")}}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
