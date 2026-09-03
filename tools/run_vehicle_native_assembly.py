"""Assemble the canonical vehicle graph in the native instrumented rig."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.abstract_ui_vehicles import _vehicle_mechanical_graph, load_default_car_configuration
from src.common.tensors import AbstractTensor
from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.common.dt_system.dt_scaler import Metrics
from src.compiler.vehicle_validator_profiles import dually_validator_profile
from src.compiler.vehicle_python_live_viewer import PythonValidatorViewer
from src.compiler.vehicle_native_assembly import (
    assembled_point_mass_properties, compile_brace_on_balance_c,
    compile_leveling_controller_c, compile_leveling_sensor_bank_c,
    compile_wheel_mesh_balance_c,
    load_vehicle_qualification_spec, native_vehicle_assembly_stages,
    qualification_stage_policy, stage_components,
)
from src.compiler.vehicle_native_deployment import derive_vehicle_rig_rate_hz
from src.compiler.vehicle_balloon_tire import balloon_tire_graph_abi
from src.compiler.vehicle_balloon_tire_program import balloon_tire_python_program
from src.compiler.vehicle_balloon_tire_diagnostics import balloon_tire_state_diagnostics
from src.compiler.vehicle_python_compilation import (
    dually_vehicle_python_compilation_inputs,
    vehicle_python_compilation_inputs, vehicle_python_runtime_bindings,
)

CORNERS = ("front_left", "front_right", "rear_left", "rear_right")
TELEMETRY_MAGIC = 8675309.0
TELEMETRY_HEADER_DOUBLES = 13


class _AssemblyTelemetry:
    """Publish the assembler-owned canonical state to the scientific viewer."""

    def __init__(self, path, vehicle_count, output_count, contact_count, fixture_count,
                 tire_state_count):
        self.path = path
        self.counts = (vehicle_count, output_count, contact_count, fixture_count,
                       tire_state_count)
        self.value_count = TELEMETRY_HEADER_DOUBLES + sum(self.counts)
        self.file = path.open("w+b")
        self.file.truncate(self.value_count * 8)
        self.mapping = mmap.mmap(self.file.fileno(), self.value_count * 8,
                                 access=mmap.ACCESS_WRITE)
        self.sequence = 0

    def publish(self, stage_index, stage_count, progress, sim_time, flags,
                vehicle_in, vehicle_out, contact_in, fixture_in, tire_state):
        self.sequence += 2
        odd = float(self.sequence - 1)
        even = float(self.sequence)
        values = [
            TELEMETRY_MAGIC, odd, 1.0, float(stage_index), float(stage_count),
            float(progress), float(sim_time), float(flags),
            *(float(count) for count in self.counts),
            *(float(value) for value in vehicle_in),
            *(float(value) for value in vehicle_out),
            *(float(value) for value in contact_in),
            *(float(value) for value in fixture_in),
            *(float(value) for value in tire_state),
        ]
        struct.pack_into("<d", self.mapping, 8, odd)
        struct.pack_into(f"<{self.value_count}d", self.mapping, 0, *values)
        struct.pack_into("<d", self.mapping, 8, even)

    def close(self):
        self.mapping.flush()
        self.mapping.close()
        self.file.close()


def _read_telemetry_snapshot(path: Path, expected_counts: tuple[int, ...]) -> dict:
    """Read one seqlock-stable native telemetry frame for exact resume."""
    raw = path.read_bytes()
    if len(raw) < TELEMETRY_HEADER_DOUBLES * 8:
        raise ValueError(f"telemetry checkpoint is truncated: {path}")
    values = struct.unpack(f"<{len(raw) // 8}d", raw)
    if values[0] != TELEMETRY_MAGIC or int(values[2]) != 1:
        raise ValueError(f"unsupported telemetry checkpoint: {path}")
    if int(values[1]) % 2:
        raise ValueError(f"telemetry checkpoint was captured during a write: {path}")
    counts = tuple(int(value) for value in values[8:13])
    if counts != expected_counts:
        raise ValueError(f"telemetry ABI mismatch: checkpoint={counts}, runtime={expected_counts}")
    offset = TELEMETRY_HEADER_DOUBLES
    arrays = []
    for count in counts:
        arrays.append(values[offset:offset + count])
        offset += count
    return {
        "stage_index": int(values[3]),
        "stage_count": int(values[4]),
        "progress": float(values[5]),
        "sim_time": float(values[6]),
        "flags": float(values[7]),
        "arrays": arrays,
    }


def _array(names, values):
    return (ctypes.c_double * len(names))(*(float(values.get(name, 0.0)) for name in names))


class _PythonCall:
    """ctypes-shaped callable used only to preserve this CLI's call sites."""

    def __init__(self, function):
        self.function = function
        self.argtypes = None

    def __call__(self, *arguments):
        return self.function(*arguments)


class _PythonVehicleMaterial:
    """Execute the authored vehicle graph in the existing pillar validator."""

    def __init__(self, vehicle_names, output_names, contact_names, fixture_names,
                 wheel_names=CORNERS, wheel_to_structural_support=None,
                 graph_constants=None, tire_dimensions=None,
                 machine_operator="configured-vehicle",
                 structural_support_positions=None,
                 tire_pneumatic_mode=None,
                 tire_material_profile="configured", prepared=None):
        self.wheel_names = tuple(wheel_names)
        if prepared is None:
            prepared = vehicle_python_compilation_inputs(
                1, self.wheel_names, wheel_to_structural_support,
                graph_constants, tire_dimensions, machine_operator,
                tuple(vehicle_names), tuple(output_names),
                structural_support_positions,
                tire_pneumatic_mode=tire_pneumatic_mode,
                tire_material_profile=tire_material_profile)
        namespace = {"AbstractTensor": AbstractTensor,
                     **vehicle_python_runtime_bindings(
                         include_configured_vehicle=(
                             machine_operator == "configured-vehicle"))}
        exec(prepared.source, namespace)
        self.entrypoint = namespace[prepared.entrypoint]
        self.energy_function = namespace["vehicle_energy_diagnostics_vector"]
        self.node_function = namespace["vehicle_material_nodes_vector"]
        eager_feeds = prepared.abstract_tensor_feeds()
        self.feed_order = tuple(eager_feeds)
        with AbstractTensor.use_backend("numpy"):
            self.feeds = {
                name: (AbstractTensor.tensor(value)
                       if isinstance(value, np.ndarray) else value)
                for name, value in eager_feeds.items()
            }
        self.vehicle_names = tuple(vehicle_names)
        self.output_names = tuple(output_names)
        self.contact_names = tuple(contact_names)
        self.fixture_names = tuple(fixture_names)
        self.vi = {name: index for index, name in enumerate(self.vehicle_names)}
        self.ci = {name: index for index, name in enumerate(self.contact_names)}
        self.fi = {name: index for index, name in enumerate(self.fixture_names)}
        tire_program = (balloon_tire_python_program(
                            self.wheel_names,
                            pneumatic_mode=tire_pneumatic_mode,
                            material_profile=tire_material_profile)
                        if tire_dimensions is None else
                        balloon_tire_python_program(
                            self.wheel_names,
                            tire_radius_m=tire_dimensions[0],
                            tire_section_radius_m=tire_dimensions[1],
                            tire_width_m=tire_dimensions[2],
                            tire_mass_kg=tire_dimensions[3],
                            reference_pressure_pa=tire_dimensions[4],
                            rim_radius_m=tire_dimensions[5],
                            pneumatic_mode=tire_pneumatic_mode,
                            material_profile=tire_material_profile))
        self.tire_faces = np.asarray(
            tire_program.constants["face_vertices"], dtype=np.int64)
        self.tire_face_zones = tuple(tire_program.face_zones)
        self.tire_face_material = np.asarray(
            tire_program.constants["face_material"], dtype=np.float64)
        self.tire_face_material_basis = np.asarray(
            tire_program.constants["face_material_basis_rad"],
            dtype=np.float64)
        self.tire_material_coordinates_uv = np.asarray(
            tire_program.constants["material_coordinates_uv"],
            dtype=np.float64)
        self.tire_rim_closure_face_mask = np.asarray(
            tire_program.constants["rim_closure_face_mask"], dtype=bool)
        self.tire_shell_surface = "single-invariant-center-surface"
        self.tire_input_index = {
            name: index for index, name in enumerate(tire_program.input_names)}
        self.last = None
        self._visual_lock = threading.Lock()
        self._visual_snapshot = None
        self._pending_visual_snapshot = None

        exported = {
            "vehicle_native_graph_tick": self.tick,
            "vehicle_native_tire_diagnostics": self.tire_diagnostics,
            "vehicle_native_tire_state": self.tire_state,
            "vehicle_native_restore_tire_state": self.restore_tire_state,
            "vehicle_native_material_state_get": self.material_state_get,
            "vehicle_native_material_state_set": self.material_state_set,
            "vehicle_native_material_diagnostics": self.material_diagnostics,
            "vehicle_native_energy_diagnostics": self.energy_diagnostics,
            "balloon_tire_contact_diagnostics": self.contact_diagnostics,
            "vehicle_native_rig_point_configure": self.configure_rig_point,
            "vehicle_native_rig_point_clear": self.clear_rig_point,
            "vehicle_native_rig_point_reactions": self.rig_reactions,
            "vehicle_native_pillar_reactions": self.pillar_reactions,
            "vehicle_native_reset": self.reset,
            "vehicle_native_set_tire_assembly": self.set_tire_assembly,
            "vehicle_native_set_tire_gas_charge": self.set_tire_gas_charge,
            "vehicle_native_set_pillar_hub_pose": self.set_pillar_pose,
            "vehicle_native_set_roller_anchor": self.set_roller_anchor,
        }
        self.exports = {name: _PythonCall(function)
                        for name, function in exported.items()}

    @staticmethod
    def _data(value):
        return np.asarray(getattr(value, "data", value))

    @staticmethod
    def _copy_flat(target, values):
        flat = np.asarray(values).reshape(-1)
        for index in range(min(len(target), len(flat))):
            target[index] = float(flat[index])

    def __getattr__(self, name):
        try:
            return self.exports[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def reset(self):
        self.last = None

    def visual_snapshot(self, *, prefer_pending=False):
        with self._visual_lock:
            source = (
                self._pending_visual_snapshot
                if prefer_pending and self._pending_visual_snapshot is not None
                else self._visual_snapshot
            )
            if source is None:
                return None
            return {name: (value.copy() if isinstance(value, np.ndarray) else value)
                    for name, value in source.items()}

    def commit_pending_visual_snapshot(self):
        with self._visual_lock:
            if self._pending_visual_snapshot is not None:
                self._visual_snapshot = self._pending_visual_snapshot
                self._pending_visual_snapshot = None

    def set_tire_assembly(self, wheel, alpha):
        self.feeds["tire_assembly_alpha"].data[0, int(wheel)] = float(alpha)

    def set_tire_gas_charge(self, charge):
        self.feeds["tire_input"].data[:, self.tire_input_index[
            "gas_charge_fraction"]] = float(charge)

    def set_pillar_pose(self, wheel, alpha, pose):
        index = int(wheel)
        self.feeds["pillar_alpha"].data[0, index] = float(alpha)
        self.feeds["pillar_pose"].data[0, index, :] = tuple(pose)

    def set_roller_anchor(self, wheel, x, z):
        index = int(wheel)
        self.feeds["roller_anchor"].data[0, index, :] = (float(x), float(z))
        self.feeds["roller_anchor_valid"].data[0, index, 0] = True

    def configure_rig_point(self, slot, mode, record):
        index = int(slot)
        row = self.feeds["rig_points"].data[0, index]
        row[:] = 0.0
        row[0], row[1] = 1.0, float(mode)
        row[2:21] = tuple(record)

    def clear_rig_point(self, slot):
        self.feeds["rig_points"].data[0, int(slot), :] = 0.0

    def tick(self, vehicle_in, contact_in, fixture_in, vehicle_out,
             publish_visual=True):
        self.feeds["vehicle_input"].data[0, :] = tuple(vehicle_in)
        validator_contact = np.asarray(tuple(contact_in)).reshape(
            (4, len(self.contact_names)))
        graph_contact = self.feeds["contact_input"].data[0]
        graph_contact[:, :] = 0.0
        graph_contact[:, 3:6] = validator_contact[:, [
            self.ci["attachment_x"], self.ci["attachment_y"],
            self.ci["attachment_z"],
        ]]
        fixture_values = np.asarray(tuple(fixture_in))
        self.feeds["fixture_global"].data[0, :] = tuple(
            fixture_values[self.fi[name]] for name in (
                "dt", "mode", "gravity", "floor_y", "carriage_mass",
                "neutral_buoyancy", "passive_damping", "lock_stiffness",
                "lock_damping", "maximum_actuator_force",
            ))
        for wheel, corner in enumerate(CORNERS[:len(self.wheel_names)]):
            self.feeds["fixture_wheel"].data[0, wheel, :] = tuple(
                (fixture_values[self.fi[f"{stem}_{corner}"]]
                 if f"{stem}_{corner}" in self.fi else 0.0) for stem in (
                    "hub_y", "hub_velocity_y", "carriage_y",
                    "carriage_velocity_y", "command_y", "command_velocity_y",
                    "roller_reaction", "mode",
                ))
        self.feeds["fixture_surface"].data[0, :] = tuple(
            fixture_values[self.fi[name]] for name in (
                "surface_mode", "terrain_phase_x", "terrain_phase_z",
                "terrain_velocity_x", "terrain_velocity_z",
                "terrain_period_x", "terrain_period_z",
            ))
        vin = self.feeds["vehicle_input"].data
        for wheel, corner in enumerate(CORNERS):
            self.feeds["wheel_assembly_alpha"].data[0, wheel] = vin[
                0, self.vi[f"assembly_alpha_{corner}"]]
            for feed_name, field in (
                ("compression", "compression"),
                ("compression_velocity", "compression_velocity"),
                ("wheel_angle", "wheel_angle"),
                ("wheel_omega", "wheel_omega"),
            ):
                self.feeds[feed_name].data[0, wheel] = vin[
                    0, self.vi[f"{field}_{corner}"]]
        self.feeds["outer_dt"].data[0] = vin[0, self.vi["dt"]]
        arguments = [self.feeds[name] for name in self.feed_order]
        with AbstractTensor.use_backend("numpy"):
            result = self.entrypoint(*arguments)
        self.last = result
        self._copy_flat(vehicle_out, self._data(result[0]))
        graph_contact_result = self._data(result[1])[0]
        for wheel in range(4):
            contact_in[wheel * len(self.contact_names) + self.ci["support"]] = (
                graph_contact_result[wheel, 6])
        corner_fixture = self._data(result[2])[0]
        for wheel, corner in enumerate(CORNERS):
            fixture_in[self.fi[f"carriage_y_{corner}"]] = corner_fixture[wheel, 0]
            fixture_in[self.fi[f"carriage_velocity_y_{corner}"]] = corner_fixture[wheel, 1]
        persistent = {
            "tire_input": result[4], "tire_state": result[5],
            "tire_output": result[6], "rig_points": self.feeds["rig_points"],
            "material_state": result[9], "roller_anchor": result[11],
            "roller_anchor_valid": result[13], "tire_initialized": result[14],
            "tire_history_valid": result[15],
        }
        history = result[7]
        persistent.update({
            "tire_previous_hub": history[0], "tire_previous_basis": history[1],
            "tire_previous_angle": history[2], "tire_previous_plane": history[3],
        })
        self.feeds.update(persistent)
        with AbstractTensor.use_backend("numpy"):
            node_position, _node_velocity = self.node_function(
                self.feeds["vehicle_input"][:, 0:3],
                self.feeds["vehicle_input"][:, 3:6],
                self.feeds["vehicle_input"][:, 6:9],
                self.feeds["vehicle_input"][:, 9:12],
                self.feeds["compression"], self.feeds["compression_velocity"],
                self.feeds["node_reference"],
                self.feeds["node_structural_support_binding"])
        snapshot = {
            "tire_position": self._data(result[5])[0, :, :, 0:3].copy(),
            "tire_faces": self.tire_faces.copy(),
            "tire_face_zones": self.tire_face_zones,
            "tire_face_material": self.tire_face_material.copy(),
            "tire_face_material_basis_rad": self.tire_face_material_basis.copy(),
            "tire_material_coordinates_uv": self.tire_material_coordinates_uv.copy(),
            "tire_rim_closure_face_mask": self.tire_rim_closure_face_mask.copy(),
            "tire_shell_surface": self.tire_shell_surface,
            "tire_pressure": self._data(result[6])[0, :, 6].copy(),
            "node_position": self._data(node_position)[0].copy(),
            "edge_nodes": self._data(self.feeds["edge_nodes"]).copy(),
            "pillar_pose": self._data(self.feeds["pillar_pose"])[0].copy(),
            "pillar_alpha": self._data(self.feeds["pillar_alpha"])[0].copy(),
            "roller_anchor": self._data(result[11])[0].copy(),
            "fixture_wheel": self._data(result[2])[0].copy(),
        }
        with self._visual_lock:
            if publish_visual:
                self._visual_snapshot = snapshot
                self._pending_visual_snapshot = None
            else:
                self._pending_visual_snapshot = snapshot

    def tire_diagnostics(self, output):
        if self.last is not None:
            self._copy_flat(output, self._data(self.last[6]))

    def tire_state(self, output):
        self._copy_flat(output, self._data(self.feeds["tire_state"]))

    def restore_tire_state(self, values):
        self.feeds["tire_state"].data[...] = np.asarray(tuple(values)).reshape(
            self.feeds["tire_state"].shape)

    def material_state_get(self, output):
        self._copy_flat(output, self._data(self.feeds["material_state"]))

    def material_state_set(self, values):
        self.feeds["material_state"].data[...] = np.asarray(tuple(values)).reshape(
            self.feeds["material_state"].shape)

    def material_diagnostics(self, output):
        if self.last is not None:
            self._copy_flat(output, self._data(self.last[10]))

    def energy_diagnostics(self, output):
        values = self.energy_function(
            self.feeds["tire_input"], self.feeds["tire_state"],
            self.feeds["tire_output"])
        self._copy_flat(output, self._data(values))

    def contact_diagnostics(self, output):
        for index in range(len(output)):
            output[index] = 0.0

    def rig_reactions(self, output):
        if self.last is not None:
            self._copy_flat(output, self._data(self.last[8]))

    def pillar_reactions(self, output):
        if self.last is not None:
            self._copy_flat(output, self._data(self.last[12]))


def _stationarity_scores(window, names, tolerances):
    """Compare adjacent vibration windows channel-by-channel.

    A periodic nonzero response is admissible when its mean and RMS envelope
    stop drifting.  The score is dimensionless; values at or below one meet
    both the absolute-plus-relative mean and vibration tolerances.
    """
    half = len(window) // 2
    first, second = window[:half], window[half:]
    scored = []
    for channel, name in enumerate(names):
        mean_a = sum(row[channel] for row in first) / half
        mean_b = sum(row[channel] for row in second) / half
        variance_a = sum((row[channel] - mean_a) ** 2 for row in first) / half
        variance_b = sum((row[channel] - mean_b) ** 2 for row in second) / half
        rms_a = math.sqrt(variance_a)
        rms_b = math.sqrt(variance_b)
        scale = max(abs(mean_a), abs(mean_b), rms_a, rms_b, 1.0)
        mean_standard_error = math.sqrt((variance_a + variance_b) / half)
        rms_standard_error = math.sqrt(
            (rms_a * rms_a + rms_b * rms_b) / max(1.0, 2.0 * (half - 1)))
        error_multiplier = float(tolerances["standard_error_multiplier"])
        mean_score = abs(mean_b - mean_a) / (
            float(tolerances["mean_absolute_floor"])
            + error_multiplier * mean_standard_error
            + float(tolerances["mean_relative_fraction"]) * scale)
        vibration_score = abs(rms_b - rms_a) / (
            float(tolerances["rms_absolute_floor"])
            + error_multiplier * rms_standard_error
            + float(tolerances["rms_relative_fraction"]) * scale)
        scored.append((max(mean_score, vibration_score), name,
                       mean_a, mean_b, rms_a, rms_b))
    return sorted(scored, reverse=True)


class _DuallyDTState:
    """Rollback-complete adapter from the validator graph to repository dt."""

    def __init__(self, material, vehicle_in, contact_in, fixture_in, vehicle_out):
        self.material = material
        self.vehicle_in = vehicle_in
        self.contact_in = contact_in
        self.fixture_in = fixture_in
        self.vehicle_out = vehicle_out

    def copy_shallow(self):
        feeds = {
            name: self.material._data(value).copy()
            for name, value in self.material.feeds.items()
            if hasattr(value, "data")
        }
        with self.material._visual_lock:
            visual = self.material._visual_snapshot
            visual = (None if visual is None else {
                name: (value.copy() if isinstance(value, np.ndarray) else value)
                for name, value in visual.items()})
            pending = self.material._pending_visual_snapshot
            pending = (None if pending is None else {
                name: (value.copy() if isinstance(value, np.ndarray) else value)
                for name, value in pending.items()})
        return (feeds, tuple(self.vehicle_in), tuple(self.contact_in),
                tuple(self.fixture_in), tuple(self.vehicle_out), visual, pending)

    def restore(self, saved):
        feeds, vehicle, contact, fixture, output, visual, pending = saved
        for name, value in feeds.items():
            self.material.feeds[name].data[...] = value
        for target, values in ((self.vehicle_in, vehicle),
                               (self.contact_in, contact),
                               (self.fixture_in, fixture),
                               (self.vehicle_out, output)):
            for index, value in enumerate(values):
                target[index] = value
        with self.material._visual_lock:
            self.material._visual_snapshot = visual
            self.material._pending_visual_snapshot = pending


def _run_dually_python_profile(args, bundle: Path, manifest: dict) -> int:
    """Run one data profile through the existing Python-authored validator."""

    if not args.python_material:
        raise ValueError("the dually profile currently requires --python-material")
    profile = dually_validator_profile()
    hz = int(manifest.get("time_integration", {}).get("outer_rate_hz", 120))
    frame_dt = 1.0 / hz
    rollback_threshold_multiplier = float(
        args.dt_rollback_threshold_multiplier)
    if rollback_threshold_multiplier < 1.0:
        raise ValueError("--dt-rollback-threshold-multiplier must be >= 1")
    vehicle_names = tuple(manifest["vehicle"]["input_names"])
    output_names = tuple(manifest["vehicle"]["output_names"])
    contact_names = tuple(manifest["contact"]["input_names"])
    fixture_names = tuple(manifest["fixture"]["input_names"])
    vi = {name: index for index, name in enumerate(vehicle_names)}
    fi = {name: index for index, name in enumerate(fixture_names)}

    defaults = {name: 0.0 for name in vehicle_names}
    mass = profile.mass_properties
    defaults.update({
        "dt": frame_dt, "position_x": 0.0, "position_y": 0.0,
        "position_z": 0.0, "gravity": 0.0, "yaw_cos": 1.0,
        "angular_damping": 0.18, "wheel_inertia": 38.0,
        "inverse_mass": 1.0 / float(mass["mass_kg"]),
        "inverse_inertia_roll": 1.0 / float(mass["inertia_kg_m2"]["roll"]),
        "inverse_inertia_pitch": 1.0 / float(mass["inertia_kg_m2"]["pitch"]),
        "inverse_inertia_yaw": 1.0 / float(mass["inertia_kg_m2"]["yaw"]),
        "assembly_alpha_drivetrain": 0.0,
        **{f"assembly_alpha_{corner}": 0.0 for corner in CORNERS},
    })
    for axis, value in zip("xyz", mass["center_of_mass"]):
        if f"center_of_mass_{axis}" in defaults:
            defaults[f"center_of_mass_{axis}"] = float(value)

    tire_radius, section_radius, _width, _tire_mass, rated_pressure, _rim_radius = (
        profile.tire_dimensions)
    contact_default = {
        "support": 1.0, "normal_y": 1.0, "forward_x": 1.0,
        "right_z": 1.0, "tire_pressure": rated_pressure,
        "tire_major_radius": tire_radius - section_radius,
        "tire_section_radius": section_radius,
    }
    contact_values = []
    for attachment in profile.wheel_attachments:
        row = dict(contact_default)
        row.update(zip(("attachment_x", "attachment_y", "attachment_z"),
                       attachment))
        contact_values.extend(float(row.get(name, 0.0))
                              for name in contact_names)

    roller_radius = 0.18
    roller_distance = math.sqrt((tire_radius + 0.13) ** 2 - roller_radius ** 2)
    carriage_y = profile.wheel_attachments[0][1] - roller_distance
    fixture_defaults = {name: 0.0 for name in fixture_names}
    fixture_defaults.update({
        "dt": frame_dt, "gravity": -9.81, "floor_y": -0.75,
        "carriage_mass": 18.0, "neutral_buoyancy": 1.0,
        "passive_damping": 12.0, "lock_stiffness": 42_000.0,
        "lock_damping": 1_800.0, "maximum_actuator_force": 32_000.0,
        "mode": 1.0, "surface_mode": 0.0,
        "terrain_period_x": 4.0, "terrain_period_z": 4.0,
    })
    for corner in CORNERS:
        fixture_defaults[f"carriage_y_{corner}"] = carriage_y
        fixture_defaults[f"command_y_{corner}"] = carriage_y
        fixture_defaults[f"mode_{corner}"] = 1.0

    prepared = dually_vehicle_python_compilation_inputs()
    material = _PythonVehicleMaterial(
        vehicle_names, output_names, contact_names, fixture_names,
        profile.wheel_names, profile.fixture_plan.wheel_to_structural_support,
        profile.graph_constants, profile.tire_dimensions,
        "structural-machine", profile.structural_support_positions,
        profile.tire_pneumatic_mode, profile.tire_material_profile,
        prepared=prepared)
    # The outer repository dt controller owns all subdivision in live mode so
    # every accepted physical substep can be published to the viewer.
    material.feeds["microstep_count"] = 1
    vehicle_in = _array(vehicle_names, defaults)
    contact_in = (ctypes.c_double * len(contact_values))(*contact_values)
    fixture_in = _array(fixture_names, fixture_defaults)
    vehicle_out = (ctypes.c_double * len(output_names))()
    pillar_pose = ctypes.c_double * 3
    for wheel, attachment in enumerate(profile.wheel_attachments):
        material.set_tire_assembly(wheel, 0.0)
        material.set_pillar_pose(wheel, 1.0, pillar_pose(*attachment))
        material.set_roller_anchor(wheel, attachment[0], attachment[2])
    ambient_charge = min(1.0, 101_325.0 / rated_pressure)
    material.set_tire_gas_charge(ambient_charge)

    state = _DuallyDTState(
        material, vehicle_in, contact_in, fixture_in, vehicle_out)
    # No live-mode floor: the repository controller may subdivide as far as
    # the authored membrane's actual error requires before retaining a step.
    controller = STController(dt_min=None, dt_max=frame_dt)
    targets = Targets(
        cfl=0.22, div_max=1.0, mass_max=1.0,
        error_limits={"maximum_substep_displacement_m": 0.006})
    stop = threading.Event()
    status_lock = threading.Condition()
    live = {
        "stage": profile.stages[0], "progress": 0.0, "sim_time": 0.0,
        "accepted_time": 0.0, "substep_dt": 0.0, "substep_index": 0,
        "accepted_substeps": 0, "rejected_substeps": 0,
        "error_max": 0.0, "error_rms": 0.0, "error_p95": 0.0,
        "error_per_wheel": (), "error_location": "none",
        "rule_violation": "none",
        "status": "starting", "finished": False, "error": None,
    }
    accepted_clock = [0.0]
    displayed_attempt = [0]
    visual_revision = [0]
    displayed_visual_revision = [0]
    stage_seconds = max(frame_dt, float(args.dually_stage_seconds))
    rig_record = ctypes.c_double * 19
    grasp_configured = False
    tire_mesh_initialized = False

    def apply_stage(stage: str, progress: float, dt_value: float) -> None:
        nonlocal grasp_configured, tire_mesh_initialized
        transfer_stage = profile.stages[7]
        wheel_alpha = (progress if stage == transfer_stage else
                       1.0 if profile.stages.index(stage) > 7 else 0.0)
        drivetrain_stage = profile.stages[5]
        drivetrain_alpha = (
            progress if stage == drivetrain_stage else
            1.0 if profile.stages.index(stage) > 5 else 0.0)
        vehicle_in[vi["assembly_alpha_drivetrain"]] = drivetrain_alpha
        for wheel, corner in enumerate(CORNERS):
            vehicle_in[vi[f"assembly_alpha_{corner}"]] = wheel_alpha
            material.feeds["wheel_assembly_alpha"].data[0, wheel] = wheel_alpha
        # The loaded wheel units arrive with their real casing state present.
        # Stage zero positions those units on their eventual-install pillars;
        # it must not gate the membrane out of the graph for an entire stage.
        # Later mounting animation changes custody/constraints, not existence.
        tire_alpha = 1.0
        for wheel in range(len(profile.wheel_names)):
            material.set_tire_assembly(wheel, tire_alpha)
        gas_charge = (ambient_charge if stage in profile.stages[:2] else
                      ambient_charge + (1.0 - ambient_charge) * progress
                      if stage == profile.stages[2] else 1.0)
        material.set_tire_gas_charge(gas_charge)
        omega = 0.65 if stage == profile.stages[3] else 0.0
        for wheel, corner in enumerate(CORNERS):
            vehicle_in[vi[f"wheel_omega_{corner}"]] = omega
            if omega:
                vehicle_in[vi[f"wheel_angle_{corner}"]] += omega * dt_value
            pillar_alpha = (1.0 - progress if stage == transfer_stage else
                            0.0 if profile.stages.index(stage) > 7 else 1.0)
            material.set_pillar_pose(
                wheel, pillar_alpha,
                pillar_pose(*profile.wheel_attachments[wheel]))
        if tire_alpha > 0.0 and not tire_mesh_initialized:
            # Establish the authored graph's exact initialized membrane at
            # zero elapsed time.  Adaptive trials now begin with a committed
            # tire frame, so rejecting the first positive-dt proposal cannot
            # roll the viewer back to the pre-installation empty state.
            vehicle_in[vi["dt"]] = 0.0
            fixture_in[fi["dt"]] = 0.0
            material.tick(vehicle_in, contact_in, fixture_in, vehicle_out,
                          publish_visual=True)
            tire_mesh_initialized = True
            with status_lock:
                visual_revision[0] += 1
                required_revision = visual_revision[0]
                live.update(
                    stage=stage,
                    progress=progress,
                    status="initialized-tire-mesh",
                )
                status_lock.notify_all()
                # The initialized graph publication is a real observable
                # frame. Do not begin the first positive-dt proposal until the
                # viewer has presented it at least once.
                while (
                    displayed_visual_revision[0] < required_revision
                    and not stop.is_set()
                ):
                    status_lock.wait(timeout=0.1)
        if stage == profile.stages[4] and not grasp_configured:
            for slot, support in enumerate(profile.structural_support_positions):
                material.configure_rig_point(slot, 1, rig_record(
                    *support, *support, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    80_000.0, 120_000.0, 80_000.0,
                    800.0, 1_200.0, 800.0, 60_000.0))
            grasp_configured = True

    def advance(_state, dt_value):
        dt_value = float(dt_value)
        attempted_time = accepted_clock[0] + dt_value
        previous = material._data(
            material.feeds["tire_state"])[0, ..., 0:3].copy()
        vehicle_in[vi["dt"]] = dt_value
        fixture_in[fi["dt"]] = dt_value
        material.tick(vehicle_in, contact_in, fixture_in, vehicle_out,
                      publish_visual=False)
        tire_state = material._data(material.feeds["tire_state"])[0]
        position = tire_state[..., 0:3]
        velocity = tire_state[..., 3:6]
        error_matrix = np.linalg.norm(position - previous, axis=-1)
        displacement = float(np.max(error_matrix))
        error_rms = float(np.sqrt(np.mean(error_matrix * error_matrix)))
        error_p95 = float(np.percentile(error_matrix, 95.0))
        error_per_wheel = tuple(float(value) for value in
                                np.max(error_matrix, axis=1).reshape(-1))
        offender = np.unravel_index(int(np.argmax(error_matrix)),
                                    error_matrix.shape)
        error_location = f"{profile.wheel_names[offender[0]]}:vertex-{offender[1]}"
        maximum_velocity = float(np.max(np.linalg.norm(velocity, axis=-1)))
        pressure = material._data(material.feeds["tire_output"])[..., 6]
        finite = bool(np.isfinite(tire_state).all() and np.isfinite(pressure).all())
        position_in_bounds = bool(np.max(np.abs(position)) < 100.0)
        pressure_below_limit = bool(np.max(pressure) < rated_pressure * 3.0)
        pressure_nonnegative = bool(np.min(pressure) >= 0.0)
        physical = bool(finite and position_in_bounds and pressure_below_limit
                        and pressure_nonnegative)
        violations = []
        if not finite:
            violations.append("finite-state rule")
        if not position_in_bounds:
            violations.append("|position| < 100 m")
        if not pressure_below_limit:
            violations.append("pressure < 3 x rated")
        if not pressure_nonnegative:
            violations.append("pressure >= 0")
        if displacement > 0.006:
            violations.append(
                f"max vertex displacement {displacement:.3e} > 6.000e-3 m")
        rule_violation = "; ".join(violations) if violations else "none"
        rollback_limit = 0.006 * rollback_threshold_multiplier
        floor_accepted = (
            controller.dt_min is not None
            and dt_value <= float(controller.dt_min) * (1.0 + 1.0e-12)
        )
        accepted = floor_accepted or (
            physical and displacement <= rollback_limit)
        soft_accepted = accepted and displacement > 0.006
        if accepted:
            material.commit_pending_visual_snapshot()
            accepted_clock[0] += dt_value
        with status_lock:
            attempt_index = int(live["substep_index"]) + 1
            live.update(
                # For a rejected attempt this is deliberately ahead of the
                # accepted clock: it is the exact candidate time just tested.
                sim_time=attempted_time,
                accepted_time=accepted_clock[0],
                substep_dt=dt_value,
                substep_index=attempt_index,
                accepted_substeps=(int(live["accepted_substeps"])
                                   + int(accepted)),
                rejected_substeps=(int(live["rejected_substeps"])
                                   + int(not accepted)),
                error_max=displacement,
                error_rms=error_rms,
                error_p95=error_p95,
                error_per_wheel=error_per_wheel,
                error_location=error_location,
                rule_violation=rule_violation,
                status=("accepted-dt-floor-substep" if floor_accepted else
                        "accepted-soft-substep" if soft_accepted else
                        "accepted-substep" if accepted
                        else "rejected-substep"))
            status_lock.notify_all()
            # Do not let a fast retry overwrite this dt before the live
            # viewer has actually presented it.
            while displayed_attempt[0] < attempt_index and not stop.is_set():
                status_lock.wait(timeout=0.1)
        if not accepted:
            print(json.dumps({
                "dt_attempt_s": dt_value,
                "attempted_time_s": attempted_time,
                "accepted_time_s": accepted_clock[0],
                "physical": physical,
                "maximum_displacement_m": displacement,
                "rms_displacement_m": error_rms,
                "p95_displacement_m": error_p95,
                "per_wheel_max_displacement_m": error_per_wheel,
                "maximum_error_location": error_location,
                "violated_rules": violations,
                "maximum_velocity_m_s": maximum_velocity,
                "pressure_pa": [float(value) for value in pressure.reshape(-1)],
            }), flush=True)
        return physical, Metrics(
            max_vel=maximum_velocity, max_flux=maximum_velocity,
            div_inf=0.0, mass_err=0.0,
            error_channels={"maximum_substep_displacement_m": displacement},
            advanced_dt=dt_value)

    def worker() -> None:
        dt_next = frame_dt
        sim_time = 0.0
        try:
            stages = profile.stages
            if args.stop_after_stage is not None:
                if args.stop_after_stage not in stages:
                    raise ValueError(
                        f"unknown dually stage {args.stop_after_stage!r}; "
                        f"expected one of {stages!r}")
                stages = stages[:stages.index(args.stop_after_stage) + 1]
            for stage in stages:
                print(json.dumps({"stage": stage, "status": "starting"}),
                      flush=True)
                elapsed = 0.0
                while elapsed < stage_seconds - 1.0e-12 and not stop.is_set():
                    window = min(1.0 / 30.0, stage_seconds - elapsed)
                    progress = min(1.0, (elapsed + window) / stage_seconds)
                    apply_stage(stage, progress, window)
                    with status_lock:
                        live.update(stage=stage, progress=progress,
                                    status="stepping")
                    advanced, dt_next, metrics = run_superstep(
                        state, window, dt_next, 0.03, targets, controller,
                        advance, allow_increase_mid_round=True,
                        max_retries=None,
                        rollback_threshold_multiplier=(
                            rollback_threshold_multiplier))
                    advanced = float(advanced)
                    if advanced <= 0.0 or metrics.hard_failure:
                        raise RuntimeError(
                            f"dt controller could not advance {stage}; "
                            f"metrics={metrics.error_channels}")
                    elapsed += advanced
                    sim_time = accepted_clock[0]
                    with status_lock:
                        live.update(stage=stage,
                                    progress=min(1.0, elapsed / stage_seconds),
                                    accepted_time=sim_time,
                                    status="running")
                print(json.dumps({"stage": stage, "status": "accepted",
                                  "sim_time_s": sim_time}), flush=True)
            with status_lock:
                live.update(finished=True, status="complete")
        except Exception as error:
            with status_lock:
                live.update(finished=True, status="failed",
                            error=traceback.format_exc())

    viewer = PythonValidatorViewer(
        profile.model, headless=args.headless_frame is not None)
    thread = threading.Thread(target=worker, name="dually-validator", daemon=True)
    thread.start()
    running = True
    try:
        while running:
            running = viewer.events()
            with status_lock:
                current = dict(live)
            # During a rejected proposal this is the exact pending graph frame,
            # held stable by the acknowledgement barrier below. Accepted
            # proposals have already promoted the same snapshot to committed.
            viewer.draw(material.visual_snapshot(prefer_pending=True),
                        stage=current["stage"],
                        progress=current["progress"],
                        sim_time=current["sim_time"], status=current["status"],
                        accepted_time=current["accepted_time"],
                        substep_dt=current["substep_dt"],
                        substep_index=current["substep_index"],
                        accepted_substeps=current["accepted_substeps"],
                        rejected_substeps=current["rejected_substeps"],
                        error_max=current["error_max"],
                        error_rms=current["error_rms"],
                        error_p95=current["error_p95"],
                        error_per_wheel=current["error_per_wheel"],
                        error_location=current["error_location"],
                        rule_violation=current["rule_violation"])
            with status_lock:
                displayed_attempt[0] = max(
                    displayed_attempt[0], int(current["substep_index"]))
                displayed_visual_revision[0] = max(
                    displayed_visual_revision[0], visual_revision[0])
                status_lock.notify_all()
            if args.headless_frame is not None and current["finished"]:
                viewer.save(args.headless_frame.resolve())
                running = False
            elif current["finished"] and not args.python_viewer:
                running = False
    finally:
        stop.set()
        with status_lock:
            status_lock.notify_all()
        thread.join(timeout=10.0)
        viewer.close()
    if live["error"]:
        raise RuntimeError(live["error"])
    print(json.dumps({
        "profile": profile.identity, "python_material": True,
        "stages": list(profile.stages), "sim_time_s": live["sim_time"],
        "wheel_count": len(profile.wheel_names),
        "pillar_count": len(profile.fixture_plan.pillars),
        "frame": (str(args.headless_frame.resolve())
                  if args.headless_frame is not None else None),
    }), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--assembly-profile", choices=("car", "dually-axle"),
                        default="car",
                        help="loaded artifact profile consumed by this validator")
    parser.add_argument("--stage-seconds", type=float, default=None,
                        help="requested ordinary-stage duration; the qualification spec may raise it to a viable observation budget")
    parser.add_argument("--wheel-seconds", type=float, default=4.0)
    parser.add_argument("--rolling-start-seconds", type=float, default=8.0)
    parser.add_argument("--release-seconds", type=float, default=8.0)
    parser.add_argument("--quiet-samples", type=int, default=64)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--viewer", action="store_true",
                        help="show the assembler-owned state in the native scientific viewer")
    parser.add_argument("--terrain-ensemble", action="store_true",
                        help="start the viewer's eight-lane exact terrain/tire batch overlay")
    parser.add_argument("--qualification-spec", type=Path, default=None,
                        help="versioned producer qualification/tolerance JSON")
    parser.add_argument("--resume-report", type=Path, default=None,
                        help="prior native assembly report whose passed prefix is retained")
    parser.add_argument("--resume-telemetry", type=Path, default=None,
                        help="matching final telemetry frame containing exact native state")
    parser.add_argument("--start-stage", default=None,
                        help="stage identity to execute first after restoring telemetry")
    parser.add_argument("--replay-checkpoint-stage", action="store_true",
                        help="explicitly allow replaying the stage that produced the checkpoint")
    parser.add_argument("--release-from-assembled-checkpoint", action="store_true",
                        help="run release directly from an exact post-leveling checkpoint, omitting the two separate drivetrain characterization stages")
    parser.add_argument("--release-new-car", action="store_true",
                        help="instantiate one fresh fully built configured car under clamps and execute only the release qualification")
    parser.add_argument("--pre-release-settle-seconds", type=float, default=20.0,
                        help="clamped contact-settle interval before clamp authority begins to ramp down")
    parser.add_argument("--fast-release-diagnostic", action="store_true",
                        help="honor short requested release duration for first-divergence debugging; never a qualification pass")
    parser.add_argument("--stop-after-stage", default=None,
                        help="write the exact post-stage checkpoint and stop")
    parser.add_argument("--python-material", action="store_true",
                        help="execute the authored Python graph instead of loading C material")
    parser.add_argument("--python-viewer", action="store_true",
                        help="show live solver tensors in the Python validator viewer")
    parser.add_argument("--headless-frame", type=Path, default=None,
                        help="render the completed Python validator state to one PNG")
    parser.add_argument("--dually-stage-seconds", type=float, default=0.25,
                        help="simulated duration assigned to each dually assembly stage")
    parser.add_argument(
        "--dt-rollback-threshold-multiplier", type=float, default=2.0,
        help=("retain a numerically over-threshold proposal for next-frame dt "
              "correction unless its error exceeds this multiple; physical "
              "and hard failures always roll back"))
    args = parser.parse_args()
    resume_requested = any((args.resume_report, args.resume_telemetry, args.start_stage))
    if resume_requested and not all((args.resume_report, args.resume_telemetry,
                                     args.start_stage)):
        parser.error("--resume-report, --resume-telemetry and --start-stage are required together")
    if args.release_new_car and resume_requested:
        parser.error("--release-new-car cannot be combined with checkpoint resume")
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "vehicle_native.manifest.json").read_text(encoding="utf-8"))
    if args.assembly_profile == "dually-axle":
        args.python_viewer = bool(args.python_viewer or args.viewer)
        if not args.python_viewer and args.headless_frame is None:
            parser.error(
                "dually-axle currently requires --python-viewer or --headless-frame")
        return _run_dually_python_profile(args, bundle, manifest)
    config = load_default_car_configuration()
    source = config.source
    qualification_spec = load_vehicle_qualification_spec(args.qualification_spec)
    quiet_tolerances = qualification_spec["quiet_tolerances"]
    contact_tolerances = qualification_spec["contact_tolerances"]
    stationarity_tolerances = qualification_spec["stationarity_tolerances"]
    observation_hz = int(qualification_spec["observation"]["sample_hz"])
    # The rig consumes the exact vehicle publication used by the game.  This
    # is intentionally not a second accessory catalogue or a copied set of
    # mount coordinates.
    mechanical_graph = _vehicle_mechanical_graph(config)
    graph_nodes = {node["identity"]: node for node in mechanical_graph["nodes"]}
    accessory_presets = mechanical_graph["rotating_accessory_presets"]
    static_accessory_presets = mechanical_graph["static_accessory_presets"]
    hz = int(manifest.get("time_integration", {}).get(
        "outer_rate_hz", derive_vehicle_rig_rate_hz(config)))
    dt = 1.0 / hz
    vehicle_names = list(manifest["vehicle"]["input_names"])
    output_names = list(manifest["vehicle"]["output_names"])
    contact_names = list(manifest["contact"]["input_names"])
    fixture_names = list(manifest["fixture"]["input_names"])
    tire_names = list(manifest["tire_appendage"]["output_names"])
    tire_stride = len(tire_names) // len(CORNERS)
    tire_state_count = int(manifest["tire_appendage"]["state_scalar_count"])
    material_state_count = int(manifest["mechanical_material"]["state_scalar_count"])
    material_diagnostic_count = int(
        manifest["mechanical_material"]["diagnostic_scalar_count"])
    vi = {name: i for i, name in enumerate(vehicle_names)}
    vo = {name: i for i, name in enumerate(output_names)}
    fi = {name: i for i, name in enumerate(fixture_names)}

    defaults = {name: 0.0 for name in vehicle_names}
    defaults.update(config.parameter_defaults())
    defaults.update({"dt": dt, "position_y": .9, "yaw_cos": 1.0,
                     "gravity": 0.0,
                     "engine_enabled": 0.0, "engine_angular_speed": 0.0, "throttle": 0.0,
                     "assembly_alpha_drivetrain": 0.0,
                     **{f"assembly_alpha_{corner}": 0.0 for corner in CORNERS}})
    tire = source["tires"]
    section = float(tire["toroid_section_radius_m"])
    contact_default = {"support": 1.0, "normal_y": 1.0, "forward_x": 1.0,
                       "right_z": 1.0, "tire_pressure": float(tire["pressure_pa"]),
                       "tire_major_radius": float(tire["radius"]) - section,
                       "tire_section_radius": section}
    wheelbase = float(source["wheels"]["wheelbase_half_length"])
    axle_offset = float(source["wheels"]["axle_group_offset_x_m"])
    track = float(source["wheels"]["track_half_width"])
    attachment_y = -float(source["chassis"]["clearance"])
    attachments = (
        (axle_offset + wheelbase, attachment_y, -track),
        (axle_offset + wheelbase, attachment_y, track),
        (axle_offset - wheelbase, attachment_y, -track),
        (axle_offset - wheelbase, attachment_y, track),
    )
    contact_values = []
    for attachment in attachments:
        row = dict(contact_default)
        row.update(zip(("attachment_x", "attachment_y", "attachment_z"), attachment))
        contact_values.extend(float(row.get(name, 0.0)) for name in contact_names)

    hub_y = float(source["suspension"]["assembly_hub_height_m"])
    roller_radius_m = .180
    bead_capture_center_distance_m = .190
    bead_capture_vertical_distance_m = math.sqrt(
        bead_capture_center_distance_m ** 2 - roller_radius_m ** 2)
    rollers_down_vertical_distance_m = math.sqrt(
        (float(tire["radius"]) + .13) ** 2 - roller_radius_m ** 2)
    carriage_y = hub_y - rollers_down_vertical_distance_m
    fixture_defaults = {name: 0.0 for name in fixture_names}
    fixture_defaults.update({"dt": dt, "gravity": -9.81, "floor_y": -.75,
        "carriage_mass": 12.0, "neutral_buoyancy": 1.0, "passive_damping": 8.0,
        "lock_stiffness": 24_000.0, "lock_damping": 1_200.0,
        "maximum_actuator_force": 18_000.0, "mode": 0.0, "surface_mode": 0.0,
        "terrain_period_x": 4.0, "terrain_period_z": 4.0})
    for corner in CORNERS:
        fixture_defaults[f"carriage_y_{corner}"] = carriage_y
        fixture_defaults[f"command_y_{corner}"] = carriage_y
        fixture_defaults[f"hub_y_{corner}"] = hub_y
        fixture_defaults[f"mode_{corner}"] = 0.0

    dll = (_PythonVehicleMaterial(
               vehicle_names, output_names, contact_names, fixture_names)
           if args.python_material
           else ctypes.CDLL(str(bundle / "vehicle_game_kernels.dll")))
    tick = dll.vehicle_native_graph_tick
    tick.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4
    tire_diagnostic = dll.vehicle_native_tire_diagnostics
    tire_diagnostic.argtypes = [ctypes.POINTER(ctypes.c_double)]
    tire_state_diagnostic = dll.vehicle_native_tire_state
    tire_state_diagnostic.argtypes = [ctypes.POINTER(ctypes.c_double)]
    restore_tire_state = dll.vehicle_native_restore_tire_state
    restore_tire_state.argtypes = [ctypes.POINTER(ctypes.c_double)]
    material_state_get = dll.vehicle_native_material_state_get
    material_state_get.argtypes = [ctypes.POINTER(ctypes.c_double)]
    material_state_set = dll.vehicle_native_material_state_set
    material_state_set.argtypes = [ctypes.POINTER(ctypes.c_double)]
    material_diagnostics = dll.vehicle_native_material_diagnostics
    material_diagnostics.argtypes = [ctypes.POINTER(ctypes.c_double)]
    energy_diagnostic = dll.vehicle_native_energy_diagnostics
    energy_diagnostic.argtypes = [ctypes.POINTER(ctypes.c_double)]
    contact_diagnostic = dll.balloon_tire_contact_diagnostics
    contact_diagnostic.argtypes = [ctypes.POINTER(ctypes.c_double)]
    configure = dll.vehicle_native_rig_point_configure
    configure.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    clear = dll.vehicle_native_rig_point_clear
    clear.argtypes = [ctypes.c_int]
    reactions = dll.vehicle_native_rig_point_reactions
    reactions.argtypes = [ctypes.POINTER(ctypes.c_double)]
    pillar_reactions = dll.vehicle_native_pillar_reactions
    pillar_reactions.argtypes = [ctypes.POINTER(ctypes.c_double)]
    dll.vehicle_native_reset()
    set_tire_assembly = dll.vehicle_native_set_tire_assembly
    set_tire_assembly.argtypes = [ctypes.c_int, ctypes.c_double]
    set_tire_gas_charge = dll.vehicle_native_set_tire_gas_charge
    set_tire_gas_charge.argtypes = [ctypes.c_double]
    set_pillar_pose = dll.vehicle_native_set_pillar_hub_pose
    set_pillar_pose.argtypes = [ctypes.c_int, ctypes.c_double,
                                ctypes.POINTER(ctypes.c_double)]
    pillar_pose = ctypes.c_double * 3
    set_roller_anchor = dll.vehicle_native_set_roller_anchor
    set_roller_anchor.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double]
    for corner_index in range(4):
        set_tire_assembly(corner_index, 0.0)
        set_pillar_pose(corner_index, 1.0, pillar_pose(
            attachments[corner_index][0], hub_y, attachments[corner_index][2]))
        set_roller_anchor(corner_index, attachments[corner_index][0],
                          attachments[corner_index][2])
    ambient_gas_charge = min(1.0, 101_325.0 / float(tire["pressure_pa"]))
    tire_gas_charge = ambient_gas_charge
    set_tire_gas_charge(tire_gas_charge)

    vehicle_in = _array(vehicle_names, defaults)
    contact_in = (ctypes.c_double * len(contact_values))(*contact_values)
    fixture_in = _array(fixture_names, fixture_defaults)
    vehicle_out = (ctypes.c_double * len(output_names))()
    tire_out = (ctypes.c_double * len(tire_names))()
    tire_state_out = (ctypes.c_double * tire_state_count)()
    material_state_out = (ctypes.c_double * material_state_count)()
    material_diagnostic_out = (ctypes.c_double * material_diagnostic_count)()
    energy_out = (ctypes.c_double * 4)()
    reaction_out = (ctypes.c_double * 96)()
    pillar_reaction_out = (ctypes.c_double * 4)()
    contact_debug = (ctypes.c_double * 16)()
    telemetry = None
    telemetry_time = 0.0
    resume_report = None
    resume_snapshot = None
    if resume_requested:
        resume_report = json.loads(args.resume_report.resolve().read_text(encoding="utf-8"))
        resume_snapshot = _read_telemetry_snapshot(
            args.resume_telemetry.resolve(),
            (len(vehicle_in), len(vehicle_out), len(contact_in), len(fixture_in),
             len(tire_state_out)),
        )
        for target, saved in zip(
                (vehicle_in, vehicle_out, contact_in, fixture_in, tire_state_out),
                resume_snapshot["arrays"]):
            for index, value in enumerate(saved):
                target[index] = value
        restore_tire_state(tire_state_out)
        material_checkpoint = args.resume_telemetry.resolve().with_suffix(".material.bin")
        material_raw = material_checkpoint.read_bytes()
        if len(material_raw) != material_state_count * 8:
            raise ValueError(
                f"material checkpoint size mismatch: {material_checkpoint}")
        for index, value in enumerate(struct.unpack(
                f"<{material_state_count}d", material_raw)):
            material_state_out[index] = value
        material_state_set(material_state_out)
        telemetry_time = resume_snapshot["sim_time"]
    if args.viewer:
        telemetry = _AssemblyTelemetry(
            bundle / "native_assembly_telemetry.bin", len(vehicle_in), len(vehicle_out),
            len(contact_in), len(fixture_in), len(tire_state_out))
        telemetry.publish(0, len(native_vehicle_assembly_stages()), 0.0, 0.0, 1.0,
                          vehicle_in, vehicle_out, contact_in, fixture_in, tire_state_out)
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        sdl_path = Path(pygame.__file__).resolve().parent / "SDL2.dll"
        if args.terrain_ensemble:
            viewer_path = bundle / "vehicle_scientific_viewer_batch.exe"
        else:
            # Prefer the threaded shell (physics thread + generation-tagged
            # snapshot double buffer) when the bundle carries it; the
            # telemetry seqlock protocol is identical in both shells.
            threaded = bundle / "vehicle_scientific_viewer_threaded.exe"
            viewer_path = (
                threaded if threaded.is_file()
                else bundle / "vehicle_scientific_viewer.exe"
            )
        if not sdl_path.is_file() or not viewer_path.is_file():
            telemetry.close()
            raise FileNotFoundError(
                f"native viewer prerequisites missing: {viewer_path}, {sdl_path}")
        viewer_command = [
            str(viewer_path), str(bundle / "vehicle_scientific.vert.glsl"),
            str(bundle / "vehicle_scientific.frag.glsl"), str(sdl_path),
            str(telemetry.path),
        ]
        if args.terrain_ensemble:
            viewer_command.append("--ensemble")
        subprocess.Popen(viewer_command, cwd=bundle)
    state_stride = tire_state_count // len(CORNERS)
    tire_vertex_count = state_stride // 6
    vibration_names = (
        [f"vehicle.{name}" for name in output_names]
        + [f"tire.{corner}.vertex_{vertex}.{component}"
           for corner in CORNERS for vertex in range(tire_vertex_count)
           for component in ("x", "y", "z", "vx", "vy", "vz")]
        + [f"tire_output.{name}" for name in tire_names]
        + [f"rig_reaction.{index}" for index in range(len(reaction_out))]
        + [f"energy.{index}" for index in range(len(energy_out))]
        + [f"contact_debug.{index}" for index in range(len(contact_debug))]
    )
    feedback = tuple((vo[name], vi[name[:-5]]) for name in output_names
                     if name.endswith("_next") and name[:-5] in vi)

    chassis = source["chassis"]
    pan_points = tuple((x, -.5 * float(chassis["height"]), z) for x, z in (
        (.92 * float(chassis["half_length"]), .92 * float(chassis["half_width"])),
        (.92 * float(chassis["half_length"]), -.92 * float(chassis["half_width"])),
        (-.92 * float(chassis["half_length"]), .92 * float(chassis["half_width"])),
        (-.92 * float(chassis["half_length"]), -.92 * float(chassis["half_width"])),
        (0.0, 0.0)))
    rig_record = ctypes.c_double * 19
    pan_targets = tuple((local[0], defaults["position_y"] + local[1], local[2])
                        for local in pan_points)
    for slot, local in enumerate(pan_points):
        target = pan_targets[slot]
        configure(slot, 1, rig_record(*local, *target, 0, 0, 0, 0, 0, 0,
            80_000, 120_000, 80_000, 800, 1_200, 800, 60_000))

    installed: list[dict] = []
    stage_reports = []
    previous_reaction = None
    failures = []
    current_properties = None
    tire_assembly_alpha = 0.0
    roller_contact_latched = [False, False, False, False]
    roller_ccd_crossed = [False, False, False, False]
    roller_contact_reason = [None, None, None, None]
    roller_locked_y = [None, None, None, None]
    roller_pressure_load_live = [False, False, False, False]
    roller_load_samples = [0, 0, 0, 0]
    roller_to_hub_clamp_distance = [math.inf, math.inf, math.inf, math.inf]
    bead_capture_distance_reached = [False, False, False, False]
    pressure_baseline_pa = [None, None, None, None]
    force_baseline_y_n = [None, None, None, None]
    pillar_alphas = [1.0, 1.0, 1.0, 1.0]
    rolling_start = {
        "starter_available": False,
        "battery_start_assist": False,
        "ignition_switched": False,
        "caught": False,
        "caught_samples": 0,
        "neutral_selected_after_catch": False,
        "maximum_engine_speed_rad_s": 0.0,
        "maximum_hub_drive_torque_nm": 0.0,
        "transfer_case_ratio": 0.0,
        "differentials_locked": False,
    }
    differential_wrench_proof = {
        "hub_mode": "locked",
        "service_brakes_locked": False,
        "maximum_open_hub_wheel_speed_rad_s": 0.0,
        "maximum_open_differential_speed_rad_s": 0.0,
        "shaft_speed_at_reconnect_rad_s": None,
        "reconnected_at_zero_slip": False,
        "maximum_locked_hub_wheel_speed_rad_s": 0.0,
        "front_and_rear_ports_driven": False,
    }
    destructive_pull = {
        "classification": "data-only-no-pass-fail",
        "terminal_event": None,
        "maximum_accessory_load_torque_nm_per_axle": 0.0,
        "maximum_clutch_slip_power_w": 0.0,
        "maximum_clutch_temperature_k": 0.0,
        "minimum_clutch_health": 1.0,
        "maximum_clutch_wear": 0.0,
        "maximum_clutch_glaze": 0.0,
        "minimum_engine_speed_rad_s": None,
        "samples": [],
    }
    leveling_program = {
        "schema": "turing.native-vehicle-leveling-program.v1",
        "loadout": [],
        "controller": {},
        "profiles": [],
        "samples": [],
    }
    leveling_corrections = {corner: 0.0 for corner in CORNERS}
    leveling_trim_corrections = {corner: 0.0 for corner in CORNERS}
    leveling_control_diagnostics = {}
    leveling_sensor_state = {}
    leveling_sensor_diagnostics = {}
    leveling_observed_support_fraction = 0.0
    chassis_clamps_released = False
    clamp_zero_samples = 0
    clamp_transfer_diagnostics = {}
    clamp_release_reference = None
    clamp_release_step = None
    clamp_release_motion = None
    spring_rate_scale = 1.0
    base_spring_rate_parameters = {
        name: float(vehicle_in[vi[name]]) for name in (
            "spring_stiffness", "spring_primary_shear_modulus_pa",
            "spring_secondary_shear_modulus_pa")
    }
    installed_accessory_inertia = {
        "external_engine_flywheel_inertia": 0.0,
        "external_differential_inertia_front": 0.0,
        "external_differential_inertia_rear": 0.0,
    }
    assembly_stages = native_vehicle_assembly_stages()
    start_stage_index = 0
    direct_release_roller_start = None
    direct_release_roller_target = None
    if args.release_new_car:
        mass = config.mass_properties()
        installed = [dict(row) for row in mass["components"]]
        current_properties = {
            "mass_kg": float(mass["total_mass_kg"]),
            "center_of_mass": tuple(float(value) for value in mass["center_of_mass"]),
            "inertia_kg_m2": {axis: float(mass["inertia_kg_m2"][axis])
                               for axis in ("roll", "pitch", "yaw")},
        }
        vehicle_in[vi["inverse_mass"]] = 1.0 / current_properties["mass_kg"]
        for axis in ("roll", "pitch", "yaw"):
            vehicle_in[vi[f"inverse_inertia_{axis}"]] = (
                1.0 / current_properties["inertia_kg_m2"][axis])
        for axis, value in zip("xyz", current_properties["center_of_mass"]):
            if f"center_of_mass_{axis}" in vi:
                vehicle_in[vi[f"center_of_mass_{axis}"]] = value
        vehicle_in[vi["assembly_alpha_drivetrain"]] = 1.0
        vehicle_in[vi["gravity"]] = 0.0
        vehicle_in[vi["engine_enabled"]] = 0.0
        vehicle_in[vi["drive_direction"]] = 0.0
        for corner_index, corner in enumerate(CORNERS):
            vehicle_in[vi[f"assembly_alpha_{corner}"]] = 1.0
            vehicle_in[vi[f"material_plastic_set_{corner}"]] = 0.0
            vehicle_in[vi[f"material_survival_{corner}"]] = 1.0
            set_tire_assembly(corner_index, 1.0)
            set_pillar_pose(corner_index, 0.0, pillar_pose(
                attachments[corner_index][0], hub_y, attachments[corner_index][2]))
            pillar_alphas[corner_index] = 0.0
        tire_assembly_alpha = 1.0
        tire_gas_charge = 1.0
        set_tire_gas_charge(tire_gas_charge)
        start_stage_index = next(index for index, stage in enumerate(assembly_stages)
                                 if stage.identity == "suspension-load-transfer")
        roller_vertical_offset = math.sqrt(
            (float(tire["radius"]) + .13) ** 2 - .18 ** 2)
        direct_release_roller_start = {}
        direct_release_roller_target = {}
        for corner in CORNERS:
            tangent_carriage_y = (defaults["position_y"] + attachment_y
                                  - roller_vertical_offset)
            direct_release_roller_start[corner] = tangent_carriage_y - .012
            direct_release_roller_target[corner] = (
                tangent_carriage_y + float(qualification_spec[
                    "clamp_release_tolerances"]["ride_height_tire_preload_m"]))
            fixture_in[fi[f"mode_{corner}"]] = 1.0
            fixture_in[fi[f"carriage_y_{corner}"]] = direct_release_roller_start[corner]
            fixture_in[fi[f"command_y_{corner}"]] = direct_release_roller_start[corner]
            fixture_in[fi[f"carriage_velocity_y_{corner}"]] = 0.0
        print(json.dumps({
            "release_new_car": True,
            "mass_kg": current_properties["mass_kg"],
            "center_of_mass": current_properties["center_of_mass"],
            "pre_release_settle_s": args.pre_release_settle_seconds,
        }), flush=True)
    if resume_requested:
        stage_identities = [stage.identity for stage in assembly_stages]
        if args.start_stage not in stage_identities:
            raise ValueError(f"unknown resume stage {args.start_stage!r}")
        start_stage_index = stage_identities.index(args.start_stage)
        completed_stage_index = int(resume_snapshot["stage_index"])
        expected_start = completed_stage_index + 1
        if start_stage_index == completed_stage_index:
            if not args.replay_checkpoint_stage:
                raise ValueError(
                    "checkpoint is post-stage state; same-stage replay requires "
                    "--replay-checkpoint-stage")
        elif (args.release_from_assembled_checkpoint
              and args.start_stage == "release"
              and stage_identities[completed_stage_index]
                  == "leveling-controller-program-capture"):
            pass
        elif start_stage_index != expected_start:
            raise ValueError(
                f"checkpoint completed stage {completed_stage_index + 1}; "
                f"the exact continuation is stage {expected_start + 1}")
        stage_reports = [dict(row) for row in resume_report.get("stages", [])
                         if stage_identities.index(row["stage"]) < start_stage_index]
        last_report = resume_report["stages"][-1]
        sensors = last_report.get("roller_contact_sensor", [])
        if len(sensors) != len(CORNERS):
            raise ValueError("resume report lacks four roller contact sensors")
        roller_ccd_crossed = [bool(row["ccd_crossed"]) for row in sensors]
        roller_contact_latched = [bool(row["pressure_load_latched"]) for row in sensors]
        roller_pressure_load_live = [bool(row["pressure_load_live"]) for row in sensors]
        roller_load_samples = [int(row["consecutive_live_load_samples"]) for row in sensors]
        roller_contact_reason = [row.get("reason") for row in sensors]
        roller_to_hub_clamp_distance = [
            float(row.get("roller_to_hub_clamp_distance_m", math.inf))
            for row in sensors]
        bead_capture_distance_reached = [
            bool(row.get("complete_bead_capture", False)) for row in sensors]
        pressure_baseline_pa = [row.get("baseline_pressure_pa") for row in sensors]
        force_baseline_y_n = [row.get("baseline_rim_force_y_n") for row in sensors]
        pillar_alphas = [float(value) for value in last_report["pillar_hub_pose_alpha"]]
        rolling_start.update(last_report.get("rolling_start", {}))
        mass = 1.0 / vehicle_in[vi["inverse_mass"]]
        current_properties = {
            "mass_kg": mass,
            "center_of_mass": tuple(
                vehicle_in[vi[f"center_of_mass_{axis}"]]
                if f"center_of_mass_{axis}" in vi else 0.0
                for axis in "xyz"),
            "inertia_kg_m2": {axis: 1.0 / vehicle_in[vi[f"inverse_inertia_{axis}"]]
                               for axis in ("roll", "pitch", "yaw")},
        }
        tire_assembly_alpha = 1.0
        for corner_index in range(4):
            set_tire_assembly(corner_index, 1.0)
            set_pillar_pose(corner_index, pillar_alphas[corner_index], pillar_pose(
                attachments[corner_index][0], hub_y, attachments[corner_index][2]))
        for coordinate in installed_accessory_inertia:
            installed_accessory_inertia[coordinate] = vehicle_in[vi[coordinate]]
        leveling_path = args.resume_report.resolve().with_name("native_leveling_program.json")
        if leveling_path.is_file():
            leveling_program = json.loads(leveling_path.read_text(encoding="utf-8"))
        print(json.dumps({"resume": args.start_stage,
                          "retained_passed_stages": len(stage_reports),
                          "telemetry_sim_time_s": telemetry_time}), flush=True)
    for stage_index, stage in enumerate(assembly_stages):
        if stage_index < start_stage_index:
            continue
        stage_failure_count = len(failures)
        if args.summary_only:
            print(f"[assembly {stage_index + 1}/{len(assembly_stages)}] {stage.identity}", flush=True)
        start_properties = current_properties
        properties = current_properties
        start_drivetrain_alpha = vehicle_in[vi["assembly_alpha_drivetrain"]]
        start_corner_alphas = tuple(vehicle_in[vi[f"assembly_alpha_{corner}"]]
                                    for corner in CORNERS)
        start_tire_alpha = tire_assembly_alpha
        start_tire_gas_charge = tire_gas_charge
        target_tire_alpha = (1.0 if stage.identity in {
            "mount-tire-casings", "inflate-tires-on-pillars", "wheel-mesh-balance", "set-suspension-rest-pose",
            "front-linkages", "rear-linkages", "armature-range-readiness",
            "gravity-admission", "rolling-start", "equipment", "accessory-installation",
            "post-accessory-ballast-balance", "leveling-controller-program-capture",
            "differential-wrench-proof",
            "destructive-drivetrain-pull", "release"} else 0.0)
        target_tire_gas_charge = (
            1.0 if stage.identity in {
                "inflate-tires-on-pillars", "wheel-mesh-balance", "set-suspension-rest-pose",
                "front-linkages", "rear-linkages", "armature-range-readiness",
                "gravity-admission", "rolling-start", "equipment", "accessory-installation",
                "post-accessory-ballast-balance", "leveling-controller-program-capture",
                "differential-wrench-proof", "destructive-drivetrain-pull", "release"
            } else ambient_gas_charge
        )
        additions = [dict(row) for row in stage_components(config, stage)
                     if row["identity"] not in {item["identity"] for item in installed}]
        if stage.identity == "accessory-installation":
            requested_accessories = (
                ("pre-clutch-crank-flywheel", "powertrain.pre_clutch_flywheel_wrench",
                 "external_engine_flywheel_inertia"),
                ("differential-port-crawl-flywheel",
                 "powertrain.front_differential_brake_wrench",
                 "external_differential_inertia_front"),
                ("differential-port-crawl-flywheel",
                 "powertrain.rear_differential_brake_wrench",
                 "external_differential_inertia_rear"),
            )
            for preset_identity, port_identity, inertia_coordinate in requested_accessories:
                parameters = accessory_presets[preset_identity]
                if port_identity not in graph_nodes:
                    raise RuntimeError(f"missing authoritative accessory port {port_identity}")
                additions.append({
                    "identity": f"installed_{preset_identity}@{port_identity}",
                    "mass_kg": float(parameters["mass_kg"]),
                    "local_position": list(graph_nodes[port_identity]["reference_position"]),
                    "mount_port": port_identity,
                    "polar_inertia_kg_m2": float(parameters["polar_inertia_kg_m2"]),
                })
                installed_accessory_inertia[inertia_coordinate] += float(
                    parameters["polar_inertia_kg_m2"])
            # The default validator deliberately carries an asymmetric gas
            # system so the following ballast stage has a real lateral moment
            # to remove.  Both devices are attached through the same
            # authoritative chassis wrench ports used by game loadouts.
            for selected in mechanical_graph["validator_default_accessory_loadout"]["items"]:
                parameters = static_accessory_presets[selected["preset"]]
                mount_ports = list(selected["mount_ports"])
                missing_ports = [port for port in mount_ports if port not in graph_nodes]
                if missing_ports:
                    raise RuntimeError(
                        f"missing authoritative accessory ports {missing_ports}")
                positions = [graph_nodes[port]["reference_position"] for port in mount_ports]
                local_position = [sum(point[axis] for point in positions) / len(positions)
                                  for axis in range(3)]
                additions.append({
                    "identity": selected["identity"],
                    "mass_kg": float(parameters["mass_kg"]),
                    "local_position": local_position,
                    "mount_ports": mount_ports,
                    "drive_port": selected.get("drive_port"),
                    "pressure_pa": float(parameters.get("reference_pressure_pa", 0.0)),
                    "temperature_k": float(parameters.get("reference_temperature_k", 0.0)),
                    "gas_mass_kg": float(parameters.get("reference_gas_mass_kg", 0.0)),
                })
            # Body selection belongs to the configuration/loadout authority.
            # Its mounts are installed with equipment; the selected physical
            # body is deliberately the final accessory so all of its mass and
            # collision-bearing structure precede the post-loadout balance.
            body_rows = [dict(row) for row in config.mass_properties()["components"]
                         if row["identity"].startswith("cosmetic_body_")]
            additions.extend(body_rows)
        installed.extend(additions)
        if installed:
            properties = assembled_point_mass_properties(tuple(installed))
            current_properties = properties

        if stage.identity in {"brace-on-balance", "post-accessory-ballast-balance"} and installed:
            artifact = compile_brace_on_balance_c()
            balance = getattr(dll, artifact.name)
            balance.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            properties = assembled_point_mass_properties(tuple(installed))
            balance_values = {"moment_x": properties["mass_kg"] * properties["center_of_mass"][0],
                              "moment_z": properties["mass_kg"] * properties["center_of_mass"][2],
                              "half_length": float(source["wheels"]["wheelbase_half_length"]),
                              "half_width": float(source["wheels"]["track_half_width"]),
                              "density": float(source["ballast"]["material_density_kg_m3"])}
            binp = _array(list(artifact.input_names), balance_values)
            bout = (ctypes.c_double * len(artifact.output_names))()
            balance(binp, bout)
            solved = dict(zip(artifact.output_names, bout))
            for corner in CORNERS:
                mass = solved[f"mass_{corner}"]
                if mass > 0:
                    prefix = ("post_accessory_solved_ballast" if
                              stage.identity == "post-accessory-ballast-balance" else
                              "solved_ballast")
                    ballast = {"identity": f"{prefix}_{corner}", "mass_kg": mass,
                               "local_position": list(pan_points[CORNERS.index(corner)])}
                    installed.append(ballast)
                    additions.append(ballast)
            properties = assembled_point_mass_properties(tuple(installed))
            current_properties = properties

        leveling_function = None
        leveling_artifact = None
        leveling_sensor_function = None
        leveling_sensor_artifact = None
        if stage.identity == "leveling-controller-program-capture":
            leveling_artifact = compile_leveling_controller_c()
            leveling_function = getattr(dll, leveling_artifact.name)
            leveling_function.argtypes = [ctypes.POINTER(ctypes.c_double),
                                          ctypes.POINTER(ctypes.c_double)]
            leveling_sensor_artifact = compile_leveling_sensor_bank_c()
            leveling_sensor_function = getattr(dll, leveling_sensor_artifact.name)
            leveling_sensor_function.argtypes = [ctypes.POINTER(ctypes.c_double),
                                                 ctypes.POINTER(ctypes.c_double)]
            leveling_program["loadout"] = [dict(row) for row in installed]
            leveling_program["controller"] = {
                "entrypoint": leveling_artifact.name,
                "input_names": list(leveling_artifact.input_names),
                "output_names": list(leveling_artifact.output_names),
                "authority": "compiler-emitted-symbolic-four-corner-load-aware-placement-law",
            }
            leveling_program["observations"] = {
                "entrypoint": leveling_sensor_artifact.name,
                "input_names": list(leveling_sensor_artifact.input_names),
                "output_names": list(leveling_sensor_artifact.output_names),
                "kind": "implicit-massless-signal-state",
                "mechanical_mass_kg": 0.0,
                "mechanical_wrench": False,
                "harness_component": False,
            }
            leveling_program["profiles"] = [
                {"identity": "level-loaded", "height": 0.0, "roll": 0.0, "pitch": 0.0},
                {"identity": "nose-up-loaded", "height": .015, "roll": 0.0, "pitch": .040},
                {"identity": "left-up-loaded", "height": .010, "roll": .035, "pitch": 0.0},
                {"identity": "level-final", "height": 0.0, "roll": 0.0, "pitch": 0.0},
            ]

        wheel_balance_rows = []
        leveling_hub_pose_errors = ([0.0] * 4 if stage.identity ==
                                    "leveling-controller-program-capture" else
                                    [math.inf] * 4)
        if stage.identity == "wheel-mesh-balance":
            artifact = compile_wheel_mesh_balance_c()
            solve_wheel = getattr(dll, artifact.name)
            solve_wheel.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            # The current steel-disc wheel is analytically axisymmetric.  A
            # generated mesh may replace these zero first moments without any
            # change to the compiled solver ABI.
            for corner_index, corner in enumerate(CORNERS):
                values = {
                    "mesh_mass": (float(source["drivetrain"]["wheel_mass_kg"])
                                  - float(source["service_lines"]["pneumatic_outer_valve_removed_hub_material_kg_each"])
                                  + float(source["service_lines"]["pneumatic_wheel_valve_mass_kg_each"])),
                    "mesh_first_moment_x": 0.0,
                    "mesh_first_moment_y": (
                        (float(source["service_lines"]["pneumatic_wheel_valve_mass_kg_each"])
                         - float(source["service_lines"]["pneumatic_outer_valve_removed_hub_material_kg_each"]))
                        * float(source["wheels"]["rim_radius"])),
                    "mesh_polar_inertia": float(vehicle_in[vi["wheel_inertia"]]),
                    "ballast_radius": float(source["wheels"]["rim_radius"]) * .92,
                    "ballast_density": float(source["ballast"]["material_density_kg_m3"]),
                    "ballast_axial_width": .025,
                    "ballast_radial_depth": .018,
                    "maximum_ballast_thickness": .040,
                }
                win = _array(list(artifact.input_names), values)
                wout = (ctypes.c_double * len(artifact.output_names))()
                solve_wheel(win, wout)
                solved = dict(zip(artifact.output_names, wout))
                mass = solved["ballast_mass"]
                if mass > 1.0e-12:
                    hub = list(attachments[corner_index])
                    hub[0] += solved["ballast_local_x"]
                    hub[1] += solved["ballast_local_y"]
                    row = {"identity": f"wheel_balance_ballast_{corner}",
                           "mass_kg": mass, "local_position": hub}
                    installed.append(row)
                    wheel_balance_rows.append(row)
                    vehicle_in[vi[f"unsprung_mass_{corner}"]] += mass
                wheel_balance_rows.append({
                    "corner": corner,
                    "mesh_source": "current-axisymmetric-wheel-definition",
                    **solved,
                })
            properties = assembled_point_mass_properties(tuple(installed))
            current_properties = properties

        quiet = 0
        stable_vibration_windows = 0
        vibration_window = []
        vibration_top_offenders = []
        monitor_samples = 0
        max_reaction = 0.0
        last_energy = None
        stage_policy = qualification_stage_policy(qualification_spec, stage.identity)
        requested_seconds = (args.release_seconds if stage.identity == "release" else
                             args.wheel_seconds if stage.identity == "inflate-tires-on-pillars"
                             else args.rolling_start_seconds if stage.identity == "rolling-start"
                             else stage.maximum_settle_seconds if args.stage_seconds is None
                             else args.stage_seconds)
        effective_seconds = (float(requested_seconds)
                             if args.fast_release_diagnostic and stage.identity == "release"
                             else max(float(requested_seconds),
                                      float(stage_policy["minimum_seconds"])))
        release_steps = max(1, round(
            min(effective_seconds, stage.maximum_settle_seconds) * hz))
        pre_release_steps = (max(0, round(args.pre_release_settle_seconds * hz))
                             if stage.identity == "release" else 0)
        steps = release_steps + pre_release_steps
        ramp_steps = max(1, steps // 2)
        release_order = (4, 0, 1, 2, 3)
        released = set(range(5)) if chassis_clamps_released else set()
        release_entry = {}
        if telemetry is not None:
            telemetry.publish(stage_index, len(assembly_stages), 0.0, telemetry_time, 1.0,
                              vehicle_in, vehicle_out, contact_in, fixture_in, tire_state_out)
        for local_step in range(steps):
            fraction = min(1.0, (local_step + 1) / ramp_steps)
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            if start_properties is None:
                live_properties = properties
            else:
                live_properties = {
                    "mass_kg": start_properties["mass_kg"] + blend * (
                        properties["mass_kg"] - start_properties["mass_kg"]),
                    "center_of_mass": tuple(
                        start_properties["center_of_mass"][axis] + blend * (
                            properties["center_of_mass"][axis]
                            - start_properties["center_of_mass"][axis]) for axis in range(3)),
                    "inertia_kg_m2": {axis: start_properties["inertia_kg_m2"][axis]
                        + blend * (properties["inertia_kg_m2"][axis]
                                   - start_properties["inertia_kg_m2"][axis])
                        for axis in ("roll", "pitch", "yaw")},
                }
            vehicle_in[vi["inverse_mass"]] = 1.0 / live_properties["mass_kg"]
            for axis in ("roll", "pitch", "yaw"):
                vehicle_in[vi[f"inverse_inertia_{axis}"]] = 1.0 / live_properties["inertia_kg_m2"][axis]
            for axis, value in zip("xyz", live_properties["center_of_mass"]):
                if f"center_of_mass_{axis}" in vi:
                    vehicle_in[vi[f"center_of_mass_{axis}"]] = value
            for coordinate, value in installed_accessory_inertia.items():
                vehicle_in[vi[coordinate]] = value
            vehicle_in[vi["assembly_alpha_drivetrain"]] = (
                start_drivetrain_alpha + blend * (stage.drivetrain_alpha - start_drivetrain_alpha))
            for corner, start_alpha, target_alpha in zip(
                    CORNERS, start_corner_alphas, stage.corner_alphas):
                vehicle_in[vi[f"assembly_alpha_{corner}"]] = (
                    start_alpha + blend * (target_alpha - start_alpha))
            transfer_stage_index = next(
                index for index, item in enumerate(assembly_stages)
                if item.identity == "suspension-load-transfer")
            gravity_target = (float(source["world"]["gravity"])
                              if stage_index >= transfer_stage_index else 0.0)
            gravity_start = float(vehicle_in[vi["gravity"]])
            if stage.identity == "suspension-load-transfer":
                vehicle_in[vi["gravity"]] = gravity_start + blend * (
                    gravity_target - gravity_start)
                vehicle_in[vi["engine_enabled"]] = 0.0
                vehicle_in[vi["drive_direction"]] = 0.0
                for corner in CORNERS:
                    vehicle_in[vi[f"external_hub_torque_{corner}"]] = 0.0
            elif stage.identity == "rolling-start":
                vehicle_in[vi["gravity"]] = gravity_target
                # L2 and the three lockers make this a slow, high-authority
                # field push.  The criterion is sustained combustion after
                # the external wrench is removed, not acceleration or speed.
                vehicle_in[vi["transfer_case_ratio"]] = float(
                    source["transmission"]["ultra_low_range_ratio"])
                for lock in ("front_differential_lock", "rear_differential_lock",
                             "center_differential_lock"):
                    vehicle_in[vi[lock]] = 1.0
                rolling_start["transfer_case_ratio"] = vehicle_in[vi["transfer_case_ratio"]]
                rolling_start["differentials_locked"] = True
                # IGN is an electrical/timing enable, not a starter.  It may
                # be switched before the push because the equation's rotation
                # gate gives it no means to create crank momentum.  START is
                # explicitly unavailable; only the hub wrench may turn the
                # connected drivetrain before combustion sustains it.
                rolling_start["ignition_switched"] = True
                vehicle_in[vi["engine_enabled"]] = float(rolling_start["ignition_switched"])
                if not rolling_start["caught"]:
                    vehicle_in[vi["drive_direction"]] = 1.0
                    target_wheel_omega = 6.5
                    for corner in CORNERS:
                        torque = max(-900.0, min(900.0, 180.0 * (
                            target_wheel_omega - vehicle_in[vi[f"wheel_omega_{corner}"]])))
                        vehicle_in[vi[f"external_hub_torque_{corner}"]] = torque
                        rolling_start["maximum_hub_drive_torque_nm"] = max(
                            rolling_start["maximum_hub_drive_torque_nm"], abs(torque))
                else:
                    vehicle_in[vi["drive_direction"]] = 0.0 if rolling_start["caught"] else 1.0
                    for corner in CORNERS:
                        vehicle_in[vi[f"external_hub_torque_{corner}"]] = 0.0
                    rolling_start["neutral_selected_after_catch"] = bool(rolling_start["caught"])
            elif stage.identity == "differential-wrench-proof":
                vehicle_in[vi["gravity"]] = gravity_target
                phase = (local_step + 1) / steps
                vehicle_in[vi["drive_direction"]] = 0.0
                for corner in CORNERS:
                    vehicle_in[vi[f"external_hub_torque_{corner}"]] = 0.0
                # Stop and hold the wheel/rotor side before operating the
                # manual clutch rings. The differential rotors remain live,
                # independently integrated angular masses on their side.
                # Reserve the final quarter of the stage for an actual
                # braked settle.  The earlier schedule drove the locked hubs
                # until 90% completion, leaving only 0.8 s for a test that
                # also required a quiescent response.
                service_brake = phase < .60 or phase >= .75
                vehicle_in[vi["brake"]] = float(service_brake)
                differential_wrench_proof["service_brakes_locked"] = service_brake
                relative_speeds = [abs(vehicle_in[vi[f"differential_wrench_shaft_omega_{axle}"]]
                                       - vehicle_in[vi[f"wheel_omega_{axle}_{side}"]])
                                   for axle in ("front", "rear") for side in ("left", "right")]
                reconnect_safe = max(relative_speeds) < .25
                open_hubs = (.10 <= phase < .50) or (phase >= .50 and not reconnect_safe)
                for corner in CORNERS:
                    vehicle_in[vi[f"hub_locker_engagement_{corner}"]] = 0.0 if open_hubs else 1.0
                differential_wrench_proof["hub_mode"] = "free" if open_hubs else "locked"
                if phase >= .50 and not open_hubs and not differential_wrench_proof[
                        "reconnected_at_zero_slip"]:
                    differential_wrench_proof["shaft_speed_at_reconnect_rad_s"] = max(
                        abs(vehicle_in[vi["differential_wrench_shaft_omega_front"]]),
                        abs(vehicle_in[vi["differential_wrench_shaft_omega_rear"]]))
                    differential_wrench_proof["reconnected_at_zero_slip"] = True
                open_shaft_drive = .20 <= phase < .35
                locked_wheel_drive = .60 <= phase < .75 and not open_hubs
                wrench_torque = 180.0 if open_shaft_drive else 140.0 if locked_wheel_drive else 0.0
                for axle in ("front", "rear"):
                    vehicle_in[vi[f"external_differential_wrench_torque_{axle}"]] = wrench_torque
                    vehicle_in[vi[f"{axle}_differential_brake"]] = float(.35 <= phase < .60
                                                                         or phase >= .75)
                differential_wrench_proof["front_and_rear_ports_driven"] = bool(
                    differential_wrench_proof["front_and_rear_ports_driven"]
                    or wrench_torque > 0.0)
            elif stage.identity == "destructive-drivetrain-pull":
                vehicle_in[vi["gravity"]] = gravity_target
                phase = (local_step + 1) / steps
                vehicle_in[vi["engine_enabled"]] = 1.0
                vehicle_in[vi["throttle"]] = 1.0
                vehicle_in[vi["drive_direction"]] = 1.0
                vehicle_in[vi["brake"]] = 0.0
                for corner in CORNERS:
                    vehicle_in[vi[f"hub_locker_engagement_{corner}"]] = 1.0
                    vehicle_in[vi[f"external_hub_torque_{corner}"]] = 0.0
                # The already-installed flywheels remain graph masses at the
                # exposed rotor-shaft ports. The absorber torque always
                # opposes measured shaft motion and ramps toward an immovable
                # load; this stage does not conjure another inertia copy.
                load_torque = 8_000.0 * min(1.0, phase / .8)
                destructive_pull["maximum_accessory_load_torque_nm_per_axle"] = max(
                    destructive_pull["maximum_accessory_load_torque_nm_per_axle"], load_torque)
                for axle in ("front", "rear"):
                    shaft_speed = vehicle_in[vi[f"differential_wrench_shaft_omega_{axle}"]]
                    vehicle_in[vi[f"external_differential_wrench_torque_{axle}"]] = (
                        -load_torque * math.tanh(shaft_speed / .5))
                    vehicle_in[vi[f"{axle}_differential_brake"]] = 0.0
            else:
                vehicle_in[vi["gravity"]] = gravity_target
                if stage.identity in {"equipment", "accessory-installation",
                                      "post-accessory-ballast-balance", "release"}:
                    vehicle_in[vi["brake"]] = 0.0
                    for axle in ("front", "rear"):
                        vehicle_in[vi[f"external_differential_wrench_torque_{axle}"]] = 0.0
                        vehicle_in[vi[f"{axle}_differential_brake"]] = 0.0
                    for corner in CORNERS:
                        vehicle_in[vi[f"hub_locker_engagement_{corner}"]] = 1.0
            tire_assembly_alpha = start_tire_alpha + blend * (
                target_tire_alpha - start_tire_alpha)
            tire_gas_charge = start_tire_gas_charge + blend * (
                target_tire_gas_charge - start_tire_gas_charge)
            set_tire_gas_charge(tire_gas_charge)
            leveling_commands = {corner: 0.0 for corner in CORNERS}
            leveling_profile = None
            if stage.identity == "leveling-controller-program-capture":
                profile_index = min(3, int(4 * local_step / max(1, steps)))
                leveling_profile = leveling_program["profiles"][profile_index]
                mode_gains = qualification_spec["leveling_tolerances"][
                    "calibrated_mode_gains"]
                falling_policy = qualification_spec["leveling_tolerances"]["falling"]
                sensor_truth = {
                    **{f"force_{corner}": tire_out[tire_stride * corner_index + 1]
                       for corner_index, corner in enumerate(CORNERS)},
                    **{f"pose_{corner}": leveling_hub_pose_errors[corner_index]
                       for corner_index, corner in enumerate(CORNERS)},
                    **{f"pressure_{corner}": tire_out[tire_stride * corner_index + 6]
                       for corner_index, corner in enumerate(CORNERS)},
                    "vertical_velocity": float(vehicle_out[vo["velocity_y_next"]]),
                }
                if not leveling_sensor_state:
                    # Initialize at physical truth so sensor start-up cannot
                    # invent a zero-load or zero-pressure transient.
                    leveling_sensor_state = dict(sensor_truth)
                sensor_values = {
                    "dt": dt,
                    "force_bandwidth_hz": float(source["suspension"][
                        "leveling_sensor_force_bandwidth_hz"]),
                    "pose_bandwidth_hz": float(source["suspension"][
                        "leveling_sensor_position_bandwidth_hz"]),
                    "pressure_bandwidth_hz": float(source["suspension"][
                        "leveling_sensor_pressure_bandwidth_hz"]),
                    "motion_bandwidth_hz": float(source["suspension"][
                        "leveling_sensor_motion_bandwidth_hz"]),
                    "force_range_n": float(source["suspension"][
                        "leveling_sensor_force_range_n"]),
                    "pose_range_m": float(source["suspension"][
                        "leveling_sensor_position_range_m"]),
                    "pressure_range_pa": float(source["suspension"][
                        "leveling_sensor_pressure_range_pa"]),
                    "motion_range_m_s": float(source["suspension"][
                        "leveling_sensor_motion_range_m_s"]),
                    **{f"truth_{name}": value for name, value in sensor_truth.items()},
                    **{f"previous_{name}": leveling_sensor_state[name]
                       for name in sensor_truth},
                }
                sinp = _array(list(leveling_sensor_artifact.input_names), sensor_values)
                sout = (ctypes.c_double * len(leveling_sensor_artifact.output_names))()
                leveling_sensor_function(sinp, sout)
                leveling_sensor_diagnostics = dict(zip(
                    leveling_sensor_artifact.output_names, sout))
                leveling_sensor_state = {
                    name: leveling_sensor_diagnostics[f"observed_{name}"]
                    for name in sensor_truth
                }
                # Pan clamps still own part of gross weight during program
                # capture. Grounded confidence is therefore the fraction of
                # four observed tire responses above the calibrated live-force
                # floor, not the fraction of GVW left in the hub fixtures.
                live_force_floor = float(contact_tolerances["minimum_live_force_n"])
                leveling_observed_support_fraction = sum(min(
                    1.0, abs(leveling_sensor_state[f"force_{corner}"])
                    / max(1e-9, live_force_floor)) for corner in CORNERS) / 4.0
                support_fraction = leveling_observed_support_fraction
                leveling_values = {
                    "target_height": leveling_profile["height"],
                    "target_roll": leveling_profile["roll"],
                    "target_pitch": leveling_profile["pitch"],
                    "target_cross_weight_correction": 0.0,
                    "half_length": float(source["wheels"]["wheelbase_half_length"]),
                    "half_width": float(source["wheels"]["track_half_width"]),
                    "corner_stiffness": float(source["suspension"]["stiffness"]),
                    "maximum_offset": float(mechanical_graph["leveling_controller"][
                        "maximum_corner_offset_m"]),
                    "dt": dt,
                    "pose_feedback_gain": float(qualification_spec[
                        "leveling_tolerances"]["pose_feedback_gain_per_s"]),
                    "trim_feedback_gain": float(qualification_spec[
                        "leveling_tolerances"]["trim_feedback_gain_per_s"]),
                    "calibrated_heave_gain": float(mode_gains["heave"]),
                    "calibrated_roll_gain": float(mode_gains["roll"]),
                    "calibrated_pitch_gain": float(mode_gains["pitch"]),
                    "calibrated_cross_weight_gain": float(mode_gains["cross_weight"]),
                    "hydraulic_pressure": float(source["suspension"][
                        "leveling_manifold_pressure_pa"]),
                    "piston_area": float(source["suspension"][
                        "leveling_actuator_piston_area_m2"]),
                    "maximum_flow": float(source["suspension"][
                        "leveling_maximum_flow_m3_s"]),
                    "hydraulic_efficiency": float(source["suspension"][
                        "leveling_hydraulic_efficiency"]),
                    "pressure_force_reserve_fraction": float(source["suspension"][
                        "leveling_pressure_force_reserve_fraction"]),
                    "coarse_rate": float(source["suspension"]["leveling_coarse_rate_m_s"]),
                    "trim_rate": float(source["suspension"]["leveling_trim_rate_m_s"]),
                    "trim_stroke": float(source["suspension"]["leveling_trim_stroke_m"]),
                    "trim_entry_error": float(source["suspension"][
                        "leveling_trim_entry_error_m"]),
                    "support_fraction": support_fraction,
                    "minimum_grounded_support_fraction": float(qualification_spec[
                        "leveling_tolerances"]["minimum_supported_weight_fraction"]),
                    "chassis_vertical_velocity": leveling_sensor_state["vertical_velocity"],
                    "fall_velocity_threshold": float(falling_policy[
                        "vertical_velocity_threshold_m_s"]),
                    "fall_velocity_blend": float(falling_policy[
                        "velocity_blend_range_m_s"]),
                    "fall_policy_selector": float(falling_policy[
                        "default_policy_selector"]),
                    "landing_ready_corner_offset": float(falling_policy[
                        "landing_ready_corner_offset_m"]),
                    "unloaded_placement_rate": float(falling_policy[
                        "maximum_unloaded_placement_rate_m_s"]),
                    "round_robin_corner": float(local_step % 4),
                    **{f"opposing_force_{corner}": abs(
                        leveling_sensor_state[f"force_{corner}"])
                       for corner in CORNERS},
                    **{f"measured_pose_error_{corner}":
                       leveling_sensor_state[f"pose_{corner}"] for corner in CORNERS},
                    **{f"previous_correction_{corner}": leveling_corrections[corner]
                       for corner in CORNERS},
                    **{f"previous_trim_{corner}": leveling_trim_corrections[corner]
                       for corner in CORNERS},
                    **{f"predicted_landing_offset_{corner}": 0.0
                       for corner in CORNERS},
                }
                lin = _array(list(leveling_artifact.input_names), leveling_values)
                lout = (ctypes.c_double * len(leveling_artifact.output_names))()
                leveling_function(lin, lout)
                leveling_result = dict(zip(leveling_artifact.output_names, lout))
                leveling_commands = {corner: leveling_result[f"command_{corner}"]
                                     for corner in CORNERS}
                leveling_corrections = {
                    corner: leveling_result[f"correction_{corner}_next"]
                    for corner in CORNERS
                }
                leveling_trim_corrections = {
                    corner: leveling_result[f"trim_{corner}_next"]
                    for corner in CORNERS
                }
                leveling_control_diagnostics = dict(leveling_result)
            for corner_index in range(4):
                set_tire_assembly(corner_index, tire_assembly_alpha)
                if stage.identity == "front-linkages" and corner_index < 2:
                    pillar_alpha = 1.0 - blend
                elif stage.identity == "rear-linkages" and corner_index >= 2:
                    pillar_alpha = 1.0 - blend
                elif stage.identity in {"rear-linkages", "suspension-load-transfer",
                                        "armature-range-readiness", "rolling-start", "equipment",
                                        "accessory-installation", "post-accessory-ballast-balance",
                                        "leveling-controller-program-capture",
                                        "differential-wrench-proof", "destructive-drivetrain-pull",
                                        "release"}:
                    pillar_alpha = 0.0 if corner_index < 2 or stage.identity != "rear-linkages" else 1.0
                else:
                    pillar_alpha = 1.0
                pillar_alphas[corner_index] = pillar_alpha
                set_pillar_pose(corner_index, pillar_alpha, pillar_pose(
                    attachments[corner_index][0],
                    hub_y + leveling_commands[CORNERS[corner_index]],
                    attachments[corner_index][2]))
            if stage.identity in {"pillar-hubs", "mount-tire-casings"}:
                # Placement happens with both articulated rollers positively
                # held at their existing lowered-clearance coordinate.
                for corner in CORNERS:
                    fixture_in[fi[f"mode_{corner}"]] = 1.0
                    fixture_in[fi[f"carriage_y_{corner}"]] = carriage_y
                    fixture_in[fi[f"command_y_{corner}"]] = carriage_y
                    fixture_in[fi[f"carriage_velocity_y_{corner}"]] = 0.0
            if stage.identity == "inflate-tires-on-pillars":
                # Assemble each balloon clear of the roller pair, then move the
                # real compiled carriages through the tire planes.  Contact is
                # therefore born from an outside-to-inside crossing, never a
                # rejection of a pre-penetrated spawn.
                for corner_index, corner in enumerate(CORNERS):
                    pressure_pa = tire_out[tire_stride * corner_index + 6]
                    rim_force_y = tire_out[tire_stride * corner_index + 1]
                    contact_count = tire_out[tire_stride * corner_index + 9]
                    volume_ratio = tire_out[tire_stride * corner_index + 7]
                    if local_step < ramp_steps:
                        fixture_in[fi[f"mode_{corner}"]] = 1.0
                        fixture_in[fi[f"carriage_y_{corner}"]] = carriage_y
                        fixture_in[fi[f"command_y_{corner}"]] = carriage_y
                        if pressure_pa > 0.0:
                            pressure_baseline_pa[corner_index] = pressure_pa
                            force_baseline_y_n[corner_index] = rim_force_y
                        fixture_in[fi[f"carriage_velocity_y_{corner}"]] = 0.0
                        continue
                    fixture_in[fi[f"mode_{corner}"]] = 0.0
                    baseline = pressure_baseline_pa[corner_index]
                    force_baseline = force_baseline_y_n[corner_index]
                    pressure_changed = (roller_ccd_crossed[corner_index]
                        and baseline is not None and
                        pressure_pa - baseline > max(
                            float(contact_tolerances["minimum_pressure_change_pa"]),
                            abs(baseline) * float(contact_tolerances[
                                "minimum_pressure_change_fraction"])))
                    force_loaded = (roller_ccd_crossed[corner_index]
                        and force_baseline is not None
                        and abs(rim_force_y - force_baseline) >= float(
                            contact_tolerances["minimum_live_force_n"]))
                    roller_pressure_load_live[corner_index] = bool(
                        pressure_changed and force_loaded and contact_count >= 1.0)
                    roller_load_samples[corner_index] = (
                        roller_load_samples[corner_index] + 1
                        if roller_pressure_load_live[corner_index] else 0
                    )
                    if contact_count >= 1.0:
                        if not roller_ccd_crossed[corner_index] and pressure_pa > 0.0:
                            pressure_baseline_pa[corner_index] = pressure_pa
                        roller_ccd_crossed[corner_index] = True
                    if volume_ratio < 0.90:
                        roller_contact_reason[corner_index] = (
                            "calibration-aborted-before-shell-fold:volume-ratio-below-0.90"
                        )
                        fixture_in[fi[f"carriage_velocity_y_{corner}"]] = 0.0
                        continue
                    hub_position_y = (vehicle_in[vi["position_y"]] - .52
                                      + vehicle_in[vi[f"compression_{corner}"]])
                    clamp_distance = hub_position_y - fixture_in[
                        fi[f"carriage_y_{corner}"]]
                    roller_to_hub_clamp_distance[corner_index] = clamp_distance
                    bead_capture_distance_reached[corner_index] = (
                        abs(clamp_distance - bead_capture_vertical_distance_m)
                        <= .0005)
                    if (roller_load_samples[corner_index] >= 8
                            and bead_capture_distance_reached[corner_index]):
                        roller_contact_latched[corner_index] = True
                        roller_contact_reason[corner_index] = "ccd-crossing+calibrated-pressure-load"
                        roller_locked_y[corner_index] = fixture_in[fi[f"carriage_y_{corner}"]]
                    if roller_contact_latched[corner_index]:
                        fixture_in[fi[f"mode_{corner}"]] = 1.0
                        fixture_in[fi[f"command_y_{corner}"]] = roller_locked_y[corner_index]
                        fixture_in[fi[f"carriage_velocity_y_{corner}"]] = 0.0
                        continue
                    old_carriage = fixture_in[fi[f"carriage_y_{corner}"]]
                    maximum_carriage = (
                        hub_position_y - bead_capture_vertical_distance_m)
                    remaining_clamp_travel = max(
                        0.0, maximum_carriage - old_carriage)
                    approach_step = min(
                        .0010, max(.00002, .20 * remaining_clamp_travel))
                    next_carriage = min(maximum_carriage, old_carriage + approach_step)
                    fixture_in[fi[f"carriage_velocity_y_{corner}"]] = (
                        next_carriage - old_carriage) / dt
                    fixture_in[fi[f"carriage_y_{corner}"]] = next_carriage
                    fixture_in[fi[f"command_y_{corner}"]] = next_carriage
            if stage.identity == "suspension-load-transfer" and not chassis_clamps_released:
                # The pedestals establish the requested ride-height geometry;
                # they are not a continuing wheel-load controller.  Once at
                # that pose they remain fixed while the actual suspension
                # spring law is scaled against the five measured clamp
                # wrenches.  This transfers real weight through tire -> hub ->
                # linkage -> spring -> chassis before any clamp is cleared.
                ride_fraction = min(1.0, fraction * 2.0)
                for corner_index, corner in enumerate(CORNERS):
                    start_y = (direct_release_roller_start or {}).get(
                        corner, fixture_in[fi[f"command_y_{corner}"]])
                    target_y = (direct_release_roller_target or {}).get(
                        corner, roller_locked_y[corner_index]
                        if roller_locked_y[corner_index] is not None else start_y)
                    fixture_in[fi[f"mode_{corner}"]] = 1.0
                    command_y = start_y + ride_fraction * (target_y - start_y)
                    fixture_in[fi[f"command_y_{corner}"]] = command_y
                if ride_fraction >= 1.0:
                    release_policy = qualification_spec["clamp_release_tolerances"]
                    clamp_vertical_reaction = sum(
                        reaction_out[6 * slot + 1] for slot in range(5))
                    gross_weight = max(1.0, properties["mass_kg"] * abs(
                        float(source["world"]["gravity"])))
                    signed_support_fraction = -clamp_vertical_reaction / gross_weight
                    exponent = max(-.02, min(.02,
                        float(release_policy["spring_rate_feedback_per_s"])
                        * signed_support_fraction * dt))
                    spring_rate_scale = max(.05, min(
                        float(release_policy["maximum_spring_rate_scale"]),
                        spring_rate_scale * math.exp(exponent)))
                    for name, base_value in base_spring_rate_parameters.items():
                        vehicle_in[vi[name]] = base_value * spring_rate_scale
            tick(vehicle_in, contact_in, fixture_in, vehicle_out)
            for output_index, input_index in feedback:
                vehicle_in[input_index] = vehicle_out[output_index]
            tire_diagnostic(tire_out)
            engine_speed = vehicle_out[vo["engine_angular_speed_next"]]
            rolling_start["maximum_engine_speed_rad_s"] = max(
                rolling_start["maximum_engine_speed_rad_s"], engine_speed)
            if stage.identity == "rolling-start" and rolling_start["ignition_switched"]:
                idle_speed = vehicle_in[vi["engine_idle_angular_speed"]]
                rolling_start["caught_samples"] = (
                    rolling_start["caught_samples"] + 1
                    if engine_speed >= .82 * idle_speed else 0)
                if rolling_start["caught_samples"] >= max(8, hz // 4):
                    rolling_start["caught"] = True
            if stage.identity == "differential-wrench-proof":
                phase = (local_step + 1) / steps
                wheel_speed = max(abs(vehicle_out[vo[f"wheel_omega_{corner}_next"]])
                                  for corner in CORNERS)
                shaft_speed = max(
                    abs(vehicle_out[vo["differential_wrench_shaft_omega_front_next"]]),
                    abs(vehicle_out[vo["differential_wrench_shaft_omega_rear_next"]]))
                if .20 <= phase < .35:
                    differential_wrench_proof["maximum_open_hub_wheel_speed_rad_s"] = max(
                        differential_wrench_proof["maximum_open_hub_wheel_speed_rad_s"],
                        wheel_speed)
                    differential_wrench_proof["maximum_open_differential_speed_rad_s"] = max(
                        differential_wrench_proof["maximum_open_differential_speed_rad_s"],
                        shaft_speed)
                if .60 <= phase < .75:
                    differential_wrench_proof["maximum_locked_hub_wheel_speed_rad_s"] = max(
                        differential_wrench_proof["maximum_locked_hub_wheel_speed_rad_s"],
                        wheel_speed)
            if stage.identity == "destructive-drivetrain-pull":
                clutch_slip_power = vehicle_out[vo["clutch_slip_power_w"]]
                clutch_temperature = vehicle_out[vo["clutch_temperature_k_next"]]
                clutch_health = vehicle_out[vo["clutch_health_next"]]
                clutch_wear = vehicle_out[vo["clutch_wear_next"]]
                clutch_glaze = vehicle_out[vo["clutch_glaze_next"]]
                destructive_pull["maximum_clutch_slip_power_w"] = max(
                    destructive_pull["maximum_clutch_slip_power_w"], clutch_slip_power)
                destructive_pull["maximum_clutch_temperature_k"] = max(
                    destructive_pull["maximum_clutch_temperature_k"], clutch_temperature)
                destructive_pull["minimum_clutch_health"] = min(
                    destructive_pull["minimum_clutch_health"], clutch_health)
                destructive_pull["maximum_clutch_wear"] = max(
                    destructive_pull["maximum_clutch_wear"], clutch_wear)
                destructive_pull["maximum_clutch_glaze"] = max(
                    destructive_pull["maximum_clutch_glaze"], clutch_glaze)
                destructive_pull["minimum_engine_speed_rad_s"] = (
                    engine_speed if destructive_pull["minimum_engine_speed_rad_s"] is None
                    else min(destructive_pull["minimum_engine_speed_rad_s"], engine_speed))
                if local_step % max(1, hz // 64) == 0:
                    destructive_pull["samples"].append({
                        "time_s": local_step * dt,
                        "engine_speed_rad_s": engine_speed,
                        "front_shaft_speed_rad_s": vehicle_out[
                            vo["differential_wrench_shaft_omega_front_next"]],
                        "rear_shaft_speed_rad_s": vehicle_out[
                            vo["differential_wrench_shaft_omega_rear_next"]],
                        "clutch_torque_nm": vehicle_out[vo["clutch_torque"]],
                        "clutch_slip_power_w": clutch_slip_power,
                        "clutch_temperature_k": clutch_temperature,
                        "clutch_health": clutch_health,
                        "clutch_wear": clutch_wear,
                        "clutch_glaze": clutch_glaze,
                        "front_differential_torque_nm": vehicle_out[
                            vo["front_differential_torque"]],
                        "rear_differential_torque_nm": vehicle_out[
                            vo["rear_differential_torque"]],
                    })
                if clutch_health <= .01 or clutch_wear >= .99:
                    destructive_pull["terminal_event"] = "clutch-thermal-wear-rupture"
                elif local_step > hz and engine_speed < .10 * vehicle_in[vi["engine_idle_angular_speed"]]:
                    destructive_pull["terminal_event"] = "engine-stall"
            # A latched historical pressure event is not live support.  Keep
            # the gate honest after every tick and through every later stage.
            for corner_index, corner in enumerate(CORNERS):
                baseline = pressure_baseline_pa[corner_index]
                force_baseline = force_baseline_y_n[corner_index]
                pressure_changed = (baseline is not None and
                    tire_out[tire_stride * corner_index + 6] - baseline
                    > max(float(contact_tolerances["minimum_pressure_change_pa"]),
                          abs(baseline) * float(contact_tolerances[
                              "minimum_pressure_change_fraction"])))
                # The five pan clamps intentionally share load until final
                # release.  A live contact must still carry a meaningful
                # fraction of its design corner load; the earlier calibration
                # latch remains the separate full-load proof.
                target_load_n = max(
                    float(contact_tolerances["minimum_live_force_n"]),
                    properties["mass_kg"] * 9.81 / 4.0
                    * float(contact_tolerances["minimum_clamped_corner_load_fraction"]),
                )
                force_loaded = (force_baseline is not None and
                    abs(tire_out[tire_stride * corner_index + 1] - force_baseline)
                    >= target_load_n)
                roller_pressure_load_live[corner_index] = bool(
                    tire_out[tire_stride * corner_index + 9] >= 1.0
                    and pressure_changed and force_loaded
                )
            energy_diagnostic(energy_out)
            reactions(reaction_out)
            pillar_reactions(pillar_reaction_out)
            contact_diagnostic(contact_debug)
            if stage.identity == "suspension-load-transfer":
                release_policy = qualification_spec["clamp_release_tolerances"]
                maximum_clamp_force = max(
                    abs(reaction_out[6 * slot + axis])
                    for slot in range(5) for axis in range(3))
                maximum_clamp_moment = max(
                    abs(reaction_out[6 * slot + 3 + axis])
                    for slot in range(5) for axis in range(3))
                maximum_compression_speed = max(
                    abs(vehicle_out[vo[f"compression_velocity_{corner}_next"]])
                    for corner in CORNERS)
                all_wheels_live = all(
                    tire_out[tire_stride * corner + 9] >= 1.0
                    for corner in range(4))
                clamp_zero_now = (
                    maximum_clamp_force <= float(release_policy["maximum_force_n"])
                    and maximum_clamp_moment <= float(release_policy["maximum_moment_nm"])
                    and maximum_compression_speed <= float(
                        release_policy["maximum_compression_speed_m_s"])
                    and all_wheels_live)
                clamp_zero_samples = clamp_zero_samples + 1 if clamp_zero_now else 0
                required_zero_samples = int(
                    release_policy["required_consecutive_zero_samples"])
                if (clamp_zero_samples >= required_zero_samples
                        and not chassis_clamps_released):
                    clamp_release_reference = {
                        "position": tuple(vehicle_out[vo[f"position_{axis}_next"]]
                                          for axis in "xyz"),
                        "velocity": tuple(vehicle_out[vo[f"velocity_{axis}_next"]]
                                          for axis in "xyz"),
                    }
                    for slot in range(5):
                        clear(slot)
                    chassis_clamps_released = True
                    clamp_release_step = local_step
                if (chassis_clamps_released and clamp_release_reference is not None
                        and clamp_release_motion is None
                        and clamp_release_step is not None
                        and local_step > clamp_release_step):
                    release_position_jump = math.sqrt(sum(
                        (vehicle_out[vo[f"position_{axis}_next"]]
                         - clamp_release_reference["position"][index]) ** 2
                        for index, axis in enumerate("xyz")))
                    release_velocity_jump = math.sqrt(sum(
                        (vehicle_out[vo[f"velocity_{axis}_next"]]
                         - clamp_release_reference["velocity"][index]) ** 2
                        for index, axis in enumerate("xyz")))
                    clamp_release_motion = (release_position_jump, release_velocity_jump)
                release_position_jump, release_velocity_jump = (
                    clamp_release_motion if clamp_release_motion is not None
                    else (None, None))
                clamp_transfer_diagnostics = {
                    "maximum_clamp_force_n": maximum_clamp_force,
                    "maximum_clamp_moment_nm": maximum_clamp_moment,
                    "maximum_compression_speed_m_s": maximum_compression_speed,
                    "consecutive_zero_samples": clamp_zero_samples,
                    "required_zero_samples": required_zero_samples,
                    "pedestal_control": "fixed-ride-height-after-preload-centering",
                    "spring_rate_scale": spring_rate_scale,
                    "bump_stop_start_compression_m": (
                        vehicle_in[vi["suspension_travel"]]
                        * vehicle_in[vi["bump_stop_start_fraction_of_travel"]]),
                    "release_position_jump_m": release_position_jump,
                    "release_velocity_jump_m_s": release_velocity_jump,
                    "all_wheels_live": all_wheels_live,
                    "chassis_clamps_released": chassis_clamps_released,
                    "release_motion_within_tolerance": bool(
                        chassis_clamps_released
                        and clamp_release_motion is not None
                        and float(release_position_jump) <= float(release_policy[
                            "maximum_release_position_jump_m"])
                        and float(release_velocity_jump) <= float(release_policy[
                            "maximum_release_velocity_jump_m_s"])),
                    "pedestal_commands_y_m": {
                        corner: fixture_in[fi[f"command_y_{corner}"]]
                        for corner in CORNERS},
                }
            if (stage.identity == "release" and pre_release_steps > 0
                    and local_step == pre_release_steps - 1):
                release_entry = {
                    "pose": {name: vehicle_out[vo[f"{name}_next"]]
                             for name in ("position_x", "position_y", "position_z",
                                          "roll", "pitch", "yaw")},
                    "velocity": {name: vehicle_out[vo[f"{name}_next"]]
                                 for name in ("velocity_x", "velocity_y", "velocity_z",
                                              "roll_velocity", "pitch_velocity",
                                              "yaw_velocity")},
                    "clamp_reactions": list(reaction_out[:30]),
                    "roller_loads_n": {
                        corner: max(0.0, tire_out[tire_stride * index + 1])
                        for index, corner in enumerate(CORNERS)},
                    "rim_forces_n": {
                        corner: list(tire_out[tire_stride * index:
                                              tire_stride * index + 3])
                        for index, corner in enumerate(CORNERS)},
                    "roller_commands_y_m": {
                        corner: fixture_in[fi[f"command_y_{corner}"]]
                        for corner in CORNERS},
                }
            if stage.identity == "leveling-controller-program-capture":
                roll = vehicle_out[vo["roll_next"]]
                pitch = vehicle_out[vo["pitch_next"]]
                yaw = vehicle_out[vo["yaw_next"]]
                cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll),
                                          math.cos(pitch), math.sin(pitch),
                                          math.cos(yaw), math.sin(yaw))
                for corner_index, corner in enumerate(CORNERS):
                    ax, ay, az = attachments[corner_index]
                    x1, z1 = cy * ax - sy * az, sy * ax + cy * az
                    y1 = cr * ay - sr * z1
                    rotated_y = -sp * x1 + cp * y1
                    articulated_y = (vehicle_out[vo["position_y_next"]] + rotated_y
                                     + vehicle_out[vo[f"compression_{corner}_next"]])
                    target_y = hub_y + leveling_commands[corner]
                    leveling_hub_pose_errors[corner_index] = articulated_y - target_y
            if (stage.identity == "leveling-controller-program-capture"
                    and local_step % max(1, hz // 64) == 0):
                leveling_program["samples"].append({
                    "time_s": local_step * dt,
                    "profile": leveling_profile["identity"],
                    "commands_m": dict(leveling_commands),
                    "coarse_corrections_m": dict(leveling_corrections),
                    "trim_corrections_m": dict(leveling_trim_corrections),
                    "control_modes": {name: leveling_result[name] for name in (
                        "heave_error", "roll_error", "pitch_error", "cross_weight_error",
                        "support_authority", "airborne_weight", "falling_weight",
                        "hydraulic_force_capacity", "hydraulic_flow_rate_limit")},
                    "sensor_maximum_normalized_residual": leveling_sensor_diagnostics[
                        "maximum_normalized_residual"],
                    "opposing_rim_force_y_n": {
                        corner: tire_out[tire_stride * corner_index + 1]
                        for corner_index, corner in enumerate(CORNERS)},
                    "body_pose": {name: vehicle_out[vo[f"{name}_next"]]
                                  for name in ("position_y", "roll", "pitch")},
                    "fixture_reaction_xyz_n": list(reaction_out[:15]),
                })
            state = tuple(vehicle_out)
            if not all(math.isfinite(value) for value in state):
                failures.append(f"{stage.identity}: non-finite state")
                break
            reaction_norm = math.sqrt(sum(reaction_out[6 * slot + axis] ** 2
                                          for slot in range(5) for axis in range(3)))
            max_reaction = max(max_reaction, reaction_norm)
            energy = tire_assembly_alpha * (energy_out[0] + energy_out[1])
            reaction_delta = math.inf if previous_reaction is None else abs(reaction_norm - previous_reaction)
            energy_delta = math.inf if last_energy is None else abs(energy - last_energy) / dt
            previous_reaction, last_energy = reaction_norm, energy
            linear_speed = math.sqrt(sum(vehicle_out[vo[f"velocity_{axis}_next"]] ** 2 for axis in "xyz"))
            angular_speed = math.sqrt(sum(vehicle_out[vo[f"{axis}_velocity_next"]] ** 2
                                          for axis in ("roll", "pitch", "yaw")))
            if (local_step >= ramp_steps
                    and linear_speed < float(quiet_tolerances["maximum_linear_speed_m_s"])
                    and angular_speed < float(quiet_tolerances["maximum_angular_speed_rad_s"])
                    and reaction_delta < float(quiet_tolerances["maximum_reaction_rate_n_s"])
                    and energy_delta < float(quiet_tolerances["maximum_energy_rate_w"])):
                quiet += 1
            else:
                quiet = 0
            # Observe every exposed native state channel at 64 Hz.  Vertex
            # velocities and the native energy channels retain the envelope
            # of modes above that observation rate, while avoiding a Python
            # copy of the full membrane on every 1024 Hz physics tick.
            if local_step >= ramp_steps and local_step % max(1, hz // observation_hz) == 0:
                tire_state_diagnostic(tire_state_out)
                snapshot = tuple(vehicle_out) + tuple(tire_state_out) + tuple(tire_out) \
                    + tuple(reaction_out) + tuple(energy_out) + tuple(contact_debug)
                state_ceiling = float(qualification_spec["hard_invariants"]["finite_absolute_state_ceiling"])
                if not all(math.isfinite(value) and abs(value) < state_ceiling for value in snapshot):
                    failures.append(f"{stage.identity}: unbounded vibration channel")
                    break
                vibration_window_length = int(stage_policy["window_samples"])
                vibration_evaluation_stride = int(stage_policy["evaluation_stride_samples"])
                vibration_window.append(snapshot)
                if len(vibration_window) > vibration_window_length:
                    vibration_window.pop(0)
                monitor_samples += 1
                if (len(vibration_window) == vibration_window_length
                        and monitor_samples % vibration_evaluation_stride == 0):
                    scores = _stationarity_scores(vibration_window, vibration_names,
                                                  stationarity_tolerances)
                    vibration_top_offenders = scores[:10]
                    if scores[0][0] <= float(stationarity_tolerances["maximum_normalized_score"]):
                        stable_vibration_windows += 1
                    else:
                        stable_vibration_windows = 0
            stage_sensor_gate = True
            if stage.identity == "inflate-tires-on-pillars":
                stage_sensor_gate = (all(roller_ccd_crossed)
                                     and all(roller_contact_latched)
                                     and all(roller_pressure_load_live)
                                     and all(bead_capture_distance_reached))
            elif stage.identity == "suspension-load-transfer":
                stage_sensor_gate = (chassis_clamps_released
                                     and clamp_transfer_diagnostics.get(
                                         "all_wheels_live", False)
                                     and clamp_transfer_diagnostics.get(
                                         "release_motion_within_tolerance", False))
            elif stage.identity == "rolling-start":
                stage_sensor_gate = (all(roller_contact_latched)
                                     and all(roller_pressure_load_live)
                                     and rolling_start["caught"]
                                     and rolling_start["neutral_selected_after_catch"])
            elif stage.identity == "differential-wrench-proof":
                open_speed_ratio = (
                    differential_wrench_proof["maximum_open_hub_wheel_speed_rad_s"]
                    / max(1.0e-12, differential_wrench_proof[
                        "maximum_open_differential_speed_rad_s"])
                )
                stage_sensor_gate = (
                    differential_wrench_proof["maximum_open_differential_speed_rad_s"] > 2.0
                    and open_speed_ratio < .05
                    and differential_wrench_proof["reconnected_at_zero_slip"]
                    and differential_wrench_proof["maximum_locked_hub_wheel_speed_rad_s"] > .5
                    and differential_wrench_proof["front_and_rear_ports_driven"])
            elif stage.identity == "leveling-controller-program-capture":
                stage_sensor_gate = (
                    max(abs(value) for value in leveling_hub_pose_errors) <= float(
                        qualification_spec["leveling_tolerances"][
                            "maximum_corner_pose_error_m"])
                    and all(roller_pressure_load_live)
                    and all(tire_out[tire_stride * corner + 9] >= 1.0
                            for corner in range(4))
                    and leveling_observed_support_fraction >= float(qualification_spec[
                        "leveling_tolerances"]["minimum_supported_corner_fraction"])
                    and all(abs(leveling_sensor_state[f"force_{corner}"]) >= float(
                        contact_tolerances["minimum_live_force_n"]) for corner in CORNERS))
            elif stage.identity == "destructive-drivetrain-pull":
                stage_sensor_gate = destructive_pull["terminal_event"] is not None
            elif stage.identity == "release":
                stage_sensor_gate = len(released) == len(release_order)
            required_quiet_samples = max(
                2, min(args.quiet_samples, stage.consecutive_quiet_samples))
            required_vibration_windows = int(stage_policy["required_stable_windows"])
            gravity_stability = (stable_vibration_windows >= 2
                                 or quiet >= max(required_quiet_samples, 2 * hz))
            stable_response = ((gravity_stability and local_step >= 3 * steps // 4)
                               if stage.identity in {"suspension-load-transfer", "rolling-start"} else
                               quiet >= required_quiet_samples
                               or stable_vibration_windows >= required_vibration_windows)
            telemetry_time += dt
            if (telemetry is not None
                    and (local_step % max(1, hz // 64) == 0 or local_step + 1 == steps)):
                tire_state_diagnostic(tire_state_out)
                telemetry.publish(stage_index, len(assembly_stages),
                                  (local_step + 1) / steps, telemetry_time, 1.0,
                                  vehicle_in, vehicle_out, contact_in, fixture_in,
                                  tire_state_out)
            if (stage.identity == "destructive-drivetrain-pull"
                    and destructive_pull["terminal_event"] is not None):
                break
            if stable_response and stage_sensor_gate:
                break
        required_quiet_samples = max(
            2, min(args.quiet_samples, stage.consecutive_quiet_samples))
        required_vibration_windows = int(stage_policy["required_stable_windows"])
        stable_response = ((stable_vibration_windows >= 2
                            or quiet >= max(required_quiet_samples, 2 * hz))
                           if stage.identity in {"suspension-load-transfer", "rolling-start"} else
                           quiet >= required_quiet_samples
                           or stable_vibration_windows >= required_vibration_windows)
        passed = len(failures) == stage_failure_count and stable_response
        if stage.identity == "destructive-drivetrain-pull":
            if destructive_pull["terminal_event"] is None:
                destructive_pull["terminal_event"] = "observation-ceiling"
            # This is characterization, not a release gate. Non-finite state
            # remains a global failure, but stall/rupture/ceiling are data.
            passed = len(failures) == stage_failure_count
        if stage.identity == "inflate-tires-on-pillars":
            passed = (passed and all(roller_ccd_crossed)
                      and all(roller_contact_latched)
                      and all(roller_pressure_load_live)
                      and all(bead_capture_distance_reached))
        if stage.identity == "gravity-admission":
            passed = (passed and all(roller_contact_latched)
                      and all(roller_pressure_load_live))
        if stage.identity == "rolling-start":
            passed = (passed and all(roller_contact_latched)
                      and all(roller_pressure_load_live)
                      and rolling_start["caught"]
                      and rolling_start["neutral_selected_after_catch"])
        if stage.identity == "differential-wrench-proof":
            open_speed_ratio = (
                differential_wrench_proof["maximum_open_hub_wheel_speed_rad_s"]
                / max(1.0e-12, differential_wrench_proof[
                    "maximum_open_differential_speed_rad_s"])
            )
            differential_wrench_proof[
                "maximum_open_hub_to_differential_speed_ratio"] = open_speed_ratio
            passed = (passed
                      and differential_wrench_proof["maximum_open_differential_speed_rad_s"] > 2.0
                      and open_speed_ratio < .05
                      and differential_wrench_proof["reconnected_at_zero_slip"]
                      and differential_wrench_proof["maximum_locked_hub_wheel_speed_rad_s"] > .5
                      and differential_wrench_proof["front_and_rear_ports_driven"])
        if stage.identity == "leveling-controller-program-capture":
            # The loop's stage_sensor_gate is the single authoritative
            # leveling acceptance rule. Recomputing a copied variant here
            # previously resurrected an obsolete pillar/GVW condition after
            # the real observed-ground-response gate had already passed.
            passed = passed and stage_sensor_gate
        if stage.identity == "release":
            released_normal_load = sum(abs(tire_out[tire_stride * corner + 1])
                                       for corner in range(4))
            passed = (passed and len(released) == len(release_order)
                      and all(tire_out[tire_stride * corner + 9] >= 1.0 for corner in range(4))
                      and released_normal_load >= float(
                          contact_tolerances["minimum_released_total_weight_fraction"]
                      ) * properties["mass_kg"] * 9.81)
        tire_modes = []
        if tire_assembly_alpha > 0.0:
            tire_state_diagnostic(tire_state_out)
            topology = balloon_tire_graph_abi(source)["topology"]
            state_stride = tire_state_count // 4
            vertex_mass = float(source["drivetrain"]["tire_mass_kg"]) / len(topology.rest_positions)
            for corner_index, corner in enumerate(CORNERS):
                wheel_state = tuple(tire_state_out[
                    corner_index * state_stride:(corner_index + 1) * state_stride
                ])
                center = [sum(wheel_state[axis::6]) / len(topology.rest_positions)
                          for axis in range(3)]
                angle = vehicle_in[vi[f"wheel_angle_{corner}"]]
                ca, sa = math.cos(angle), math.sin(angle)
                reference = []
                for x, y, z in topology.rest_positions:
                    reference.extend((center[0] + ca*x - sa*y,
                                      center[1] + sa*x + ca*y,
                                      center[2] + z, 0.0, 0.0, 0.0))
                tire_modes.append({"corner": corner, **balloon_tire_state_diagnostics(
                    wheel_state, reference, topology, vertex_mass,
                )})
        material_diagnostics(material_diagnostic_out)
        material_rows = [material_diagnostic_out[index:index + 8]
                         for index in range(0, material_diagnostic_count, 8)]
        stage_report = {"stage": stage.identity, "pass": passed,
            "installed": [row["identity"] for row in additions], "installed_mass_kg": properties["mass_kg"],
            "maximum_clamp_reaction_n": max_reaction, "quiet_samples": quiet,
            "stable_vibration_windows": stable_vibration_windows,
            "stability_gate": ("stationary-all-channel-vibration"
                               if stable_vibration_windows >= required_vibration_windows else
                               "quiescent-observation-only" if quiet >= required_quiet_samples
                               else "not-stable"),
            "gate_results": {
                "stability": bool(stable_response),
                "stage_sensor": bool(stage_sensor_gate),
            },
            "qualification_spec": qualification_spec["identity"],
            "requested_duration_s": requested_seconds,
            "effective_duration_ceiling_s": min(effective_seconds, stage.maximum_settle_seconds),
            "vibration_observation_hz": observation_hz,
            "rolling_start": dict(rolling_start),
            "differential_wrench_proof": dict(differential_wrench_proof),
            "destructive_drivetrain_pull": dict(destructive_pull),
            "vibration_top_offenders": [{"score": row[0], "channel": row[1],
                                           "mean_previous": row[2], "mean_current": row[3],
                                           "rms_previous": row[4], "rms_current": row[5]}
                                          for row in vibration_top_offenders],
            "gravity_m_s2": vehicle_in[vi["gravity"]],
            "pillar_hub_pose_alpha": list(pillar_alphas),
            "wheel_mesh_balance": wheel_balance_rows,
            "leveling_hub_pose_error_m": dict(zip(CORNERS, leveling_hub_pose_errors)),
            "leveling_control_diagnostics": leveling_control_diagnostics,
            "leveling_observations": ({
                "kind": "implicit-massless-signal-state",
                "truth": {
                    **{f"force_{corner}": tire_out[tire_stride * corner_index + 1]
                       for corner_index, corner in enumerate(CORNERS)},
                    **{f"pose_{corner}": leveling_hub_pose_errors[corner_index]
                       for corner_index, corner in enumerate(CORNERS)},
                    **{f"pressure_{corner}": tire_out[tire_stride * corner_index + 6]
                       for corner_index, corner in enumerate(CORNERS)},
                    "vertical_velocity": float(vehicle_out[vo["velocity_y_next"]]),
                },
                "observed": dict(leveling_sensor_state),
                "diagnostics": dict(leveling_sensor_diagnostics),
                "observed_supported_corner_fraction": leveling_observed_support_fraction,
                "support_definition": "four-observed-tire-ground-responses-above-live-force-floor",
            } if stage.identity == "leveling-controller-program-capture" else {}),
            "pillar_reaction_y_n": dict(zip(CORNERS, pillar_reaction_out)),
            "wheel_force_sensors": [{
                "corner": CORNERS[corner],
                "rim_force_n": [tire_out[tire_stride * corner + axis] for axis in range(3)],
                "rim_moment_nm": [tire_out[tire_stride * corner + 3 + axis] for axis in range(3)],
                "roller_normal_load_n": vehicle_in[
                    vi[f"contact_normal_force_{CORNERS[corner]}"]],
                "pressure_pa": tire_out[tire_stride * corner + 6],
            } for corner in range(4)],
            "tire_manifold_modes": tire_modes,
            "active_tire_contact_vertices": tire_assembly_alpha * sum(
                tire_out[tire_stride * corner + 9] for corner in range(4)),
            "contact_active_band_m": .02 * float(source["tire_skin"]["skin_thickness_m"]),
            "minimum_tire_skin_y_m": min(
                tire_out[tire_stride * corner + 10] for corner in range(4)),
            "contact_ccd": [{"minimum_previous_distance_m": contact_debug[4 * corner],
                             "minimum_current_distance_m": contact_debug[4 * corner + 1],
                             "crossing_candidates": contact_debug[4 * corner + 2],
                             "inside_triangle_candidates": contact_debug[4 * corner + 3]}
                            for corner in range(4)],
            "roller_contact_sensor": [{"ccd_crossed": roller_ccd_crossed[corner],
                                       "pressure_load_latched": roller_contact_latched[corner],
                                       "pressure_load_live": roller_pressure_load_live[corner],
                                       "consecutive_live_load_samples": roller_load_samples[corner],
                                       "latched": roller_contact_latched[corner],
                                       "reason": roller_contact_reason[corner],
                                       "roller_to_hub_clamp_distance_m": (
                                           roller_to_hub_clamp_distance[corner]),
                                       "target_bead_capture_distance_m": (
                                           bead_capture_vertical_distance_m),
                                       "complete_bead_capture": (
                                           bead_capture_distance_reached[corner]),
                                       "baseline_pressure_pa": pressure_baseline_pa[corner],
                                       "baseline_rim_force_y_n": force_baseline_y_n[corner],
                                           "pressure_delta_pa": (None if pressure_baseline_pa[corner] is None
                                               else tire_out[tire_stride * corner + 6]
                                                    - pressure_baseline_pa[corner])}
                                      for corner in range(4)],
            "final_pose": {name: vehicle_out[vo[f"{name}_next"]]
                           for name in ("position_x", "position_y", "position_z",
                                        "roll", "pitch", "yaw")},
            "final_velocity": {name: vehicle_out[vo[f"{name}_next"]]
                               for name in ("velocity_x", "velocity_y", "velocity_z",
                                            "roll_velocity", "pitch_velocity", "yaw_velocity")},
            "mechanical_material": {
                "edge_count": len(material_rows),
                "failed_edge_count": sum(row[4] >= .5 for row in material_rows),
                "plastic_edge_count": sum(abs(row[1]) > 1.0e-12 for row in material_rows),
                "maximum_accumulated_plastic_strain": max(
                    (row[2] for row in material_rows), default=0.0),
                "maximum_fracture_demand": max(
                    (row[6] for row in material_rows), default=0.0),
                "total_dissipated_energy_j": sum(row[7] for row in material_rows),
            },
            "release_entry": release_entry}
        stage_report["clamp_load_transfer"] = dict(clamp_transfer_diagnostics)
        stage_reports.append(stage_report)
        # Preserve exact post-stage state before any later excitation can
        # overwrite it.  A separate file per stage makes clean release and
        # targeted recovery trials possible without replaying assembly.
        checkpoint_stem = f"native_assembly_checkpoint_{stage_index + 1:02d}_{stage.identity}"
        checkpoint_binary = bundle / f"{checkpoint_stem}.bin"
        checkpoint_report = bundle / f"{checkpoint_stem}.json"
        tire_state_diagnostic(tire_state_out)
        checkpoint = _AssemblyTelemetry(
            checkpoint_binary, len(vehicle_in), len(vehicle_out), len(contact_in),
            len(fixture_in), len(tire_state_out))
        checkpoint.publish(stage_index, len(assembly_stages), 1.0, telemetry_time,
                           2.0 if passed else 4.0, vehicle_in, vehicle_out,
                           contact_in, fixture_in, tire_state_out)
        checkpoint.close()
        material_state_get(material_state_out)
        material_checkpoint = checkpoint_binary.with_suffix(".material.bin")
        material_checkpoint.write_bytes(struct.pack(
            f"<{material_state_count}d", *material_state_out))
        stage_report["checkpoint"] = {
            "telemetry": checkpoint_binary.name,
            "material_state": material_checkpoint.name,
            "report": checkpoint_report.name,
            "state_semantics": "exact-post-stage-including-persistent-material-history",
        }
        checkpoint_report.write_text(json.dumps({
            "schema": "turing.native-vehicle-assembly-checkpoint.v1",
            "completed_stage_index": stage_index,
            "completed_stage": stage.identity,
            "next_stage": (assembly_stages[stage_index + 1].identity
                           if stage_index + 1 < len(assembly_stages) else None),
            "qualification_spec": qualification_spec["identity"],
            "physics_hz": hz,
            "stages": stage_reports,
        }, indent=2), encoding="utf-8")
        if not args.summary_only:
            print(json.dumps(stage_report), flush=True)
        else:
            print(json.dumps({"stage": stage.identity, "pass": passed,
                              "gate": stage_report["stability_gate"],
                              "quiet_samples": quiet,
                              "stable_vibration_windows": stable_vibration_windows,
                              "top_offender": (vibration_top_offenders[0][1]
                                               if vibration_top_offenders else None)}),
                  flush=True)
        if not passed:
            failed_gates = []
            if not stable_response:
                failed_gates.append("stability")
            if not stage_sensor_gate:
                failed_gates.append("stage-sensor")
            failures.append(
                f"{stage.identity}: failed {','.join(failed_gates) or 'stage-specific'} gate"
            )
            # Preserve the live assembled state and continue.  Late-stage
            # characterization and release remain valuable even when an
            # earlier qualification gate fails, and the final report retains
            # every failure without forcing another stages-1..N replay.
        if args.stop_after_stage == stage.identity:
            break
        if stage.identity == "suspension-load-transfer" and not passed:
            break

    result = {"schema": "turing.native-vehicle-assembly-run.v1", "pass": not failures,
              "qualification_spec": qualification_spec["identity"],
              "physics_hz": hz, "failures": failures, "stages": stage_reports}
    destination = bundle / "native_assembly_report.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (bundle / "native_leveling_program.json").write_text(
        json.dumps(leveling_program, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(destination), "pass": result["pass"],
                      "failures": failures}, indent=2))
    if telemetry is not None:
        telemetry.publish(max(0, len(stage_reports) - 1), len(assembly_stages), 1.0,
                          telemetry_time, 2.0 if result["pass"] else 4.0,
                          vehicle_in, vehicle_out, contact_in, fixture_in, tire_state_out)
        telemetry.close()
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
