"""Repeatable exact-teacher trials for the balloon-tire hub-wrench operator.

This module deliberately owns no contact or constitutive equation.  It calls
the content-addressed native tire authority and only supplies rig states,
collects its bead/rim boundary wrench, and packs those states for the learned
operator.  It is an identification rig, not a second tire implementation.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .vehicle_tire_authority import WrittenNativeTireAuthority
from .vehicle_tire_force_network import (
    HUB_WRENCH_NAMES,
    MEMBRANE_FIELD_NAMES,
    TIRE_OPERATOR_OUTPUT_NAMES,
    NativeTireForceOperator,
    TireForceNetworkSpec,
    TireForceNetworkTrainer,
    TireForceTrainingGraphs,
    pack_tire_force_features,
    reduce_teacher_bead_wrenches,
)
from .ssa_llvm_backend import prepare_artifact_execution


@dataclass(frozen=True, slots=True)
class TireTrialBatch:
    features: np.ndarray
    target_operator_output: np.ndarray
    membrane_history: np.ndarray
    terrain_history: np.ndarray
    vehicle_state: np.ndarray
    teacher_input: np.ndarray | None = None
    teacher_state: np.ndarray | None = None
    target_membrane_state: np.ndarray | None = None

    @property
    def target_hub_wrench(self) -> np.ndarray:
        return self.target_operator_output[:, :len(HUB_WRENCH_NAMES)]

    @property
    def target_thermodynamic_state(self) -> np.ndarray:
        return self.target_operator_output[:, len(HUB_WRENCH_NAMES):]


@dataclass(frozen=True, slots=True)
class LinearTireForceOperator:
    """One affine full-state map; deploys as a single augmented GPU GEMM."""

    weights: np.ndarray
    feature_count: int
    ridge: float

    def predict(self, features: np.ndarray, spec: TireForceNetworkSpec) -> np.ndarray:
        matrix = tire_feature_matrix(features, spec)
        augmented = np.concatenate((matrix, np.ones((len(matrix), 1))), axis=1)
        return (augmented @ self.weights) * spec.output_scale


@dataclass(frozen=True, slots=True)
class LinearTireStateOperator:
    """Shared per-vertex affine next-state map for GPU-resident propagation."""

    weights: np.ndarray
    input_channels: int
    ridge: float

    @staticmethod
    def output_scale() -> np.ndarray:
        return np.asarray([1.0, 1.0, 1.0, 30.0, 30.0, 30.0], dtype=np.float64)

    def predict(self, features: np.ndarray, spec: TireForceNetworkSpec) -> np.ndarray:
        halo = spec.halo
        core = np.asarray(features, dtype=np.float64)[
            :, :, halo:halo + spec.circumferential_segments,
            halo:halo + spec.section_segments,
        ]
        matrix = np.moveaxis(core, 1, -1).reshape(-1, spec.input_channels)
        augmented = np.concatenate((matrix, np.ones((len(matrix), 1))), axis=1)
        predicted = (augmented @ self.weights) * self.output_scale()
        return predicted.reshape(
            spec.batch_size, spec.circumferential_segments,
            spec.section_segments, len(MEMBRANE_FIELD_NAMES),
        )


class CompiledTireTeacher:
    """ctypes adapter over the compiler-emitted scalar authority DLL."""

    def __init__(self, written: WrittenNativeTireAuthority):
        self.written = written
        self.manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
        self.library = ctypes.CDLL(str(written.library_path))
        self._functions: dict[str, Any] = {}
        for name in self.manifest["native"]["abi"]:
            function = getattr(self.library, name)
            function.restype = None
            function.argtypes = [
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ]
            self._functions[name] = function
        appendage = self.manifest["native"].get("appendage")
        self.appendage = appendage
        if appendage is not None:
            self._appendage_defaults = getattr(self.library, appendage["defaults"])
            self._appendage_initialize = getattr(self.library, appendage["initialize"])
            self._appendage_step = getattr(self.library, appendage["step"])
            pointer = ctypes.POINTER(ctypes.c_double)
            self._appendage_defaults.argtypes = [pointer]
            self._appendage_initialize.argtypes = [pointer, pointer]
            self._appendage_step.argtypes = [pointer, pointer, pointer]

    def appendage_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.appendage is None:
            raise RuntimeError("tire authority lacks the whole balloon appendage ABI")
        inputs = np.zeros(len(self.appendage["inputs"]), dtype=np.float64)
        state = np.zeros(int(self.appendage["state_scalar_count"]), dtype=np.float64)
        outputs = np.zeros(len(self.appendage["outputs"]), dtype=np.float64)
        self._appendage_defaults(inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return inputs, state, outputs

    def initialize_appendage(self, inputs: np.ndarray, state: np.ndarray) -> None:
        self._appendage_initialize(
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def step_appendage(
        self, inputs: np.ndarray, state: np.ndarray, outputs: np.ndarray,
    ) -> None:
        self._appendage_step(
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def call(self, name: str, values: Mapping[str, float]) -> dict[str, float]:
        abi = self.manifest["native"]["abi"][name]
        inputs = np.asarray([values[item] for item in abi["inputs"]], dtype=np.float64)
        outputs = np.empty(len(abi["outputs"]), dtype=np.float64)
        self._functions[name](
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return dict(zip(abi["outputs"], map(float, outputs), strict=True))

    def bead_wrench(
        self,
        *,
        positions: np.ndarray,
        velocities: np.ndarray,
        targets: np.ndarray,
        target_velocities: np.ndarray,
        rim_center: np.ndarray,
        stiffness: float,
        damping: float,
    ) -> np.ndarray:
        forces, moments = [], []
        for position, velocity, target, target_velocity in zip(
            positions, velocities, targets, target_velocities, strict=True,
        ):
            result = self.call("balloon_tire_bead_implicit_step", {
                "dt": 1.0 / 65536.0,
                "vertex_mass_kg": float(self.manifest["runtime_parameters"]["vertex_mass_kg"]),
                "bead_damping_n_s_per_m": damping,
                "bead_stiffness_n_per_m": stiffness,
                **{f"rim_center_{axis}": rim_center[i] for i, axis in enumerate("xyz")},
                **{f"target_velocity_{axis}": target_velocity[i] for i, axis in enumerate("xyz")},
                **{f"target_{axis}": target[i] for i, axis in enumerate("xyz")},
                **{f"free_velocity_{axis}": velocity[i] for i, axis in enumerate("xyz")},
                **{f"vertex_{axis}": position[i] for i, axis in enumerate("xyz")},
            })
            forces.append([result[f"rim_force_{axis}_n"] for axis in "xyz"])
            moments.append([result[f"rim_moment_{axis}_nm"] for axis in "xyz"])
        return reduce_teacher_bead_wrenches(
            np.asarray(forces)[None, ...], np.asarray(moments)[None, ...],
        )[0]


class TireTrialGenerator:
    """Build smooth rig excitations and label them through the native teacher."""

    def __init__(
        self,
        teacher: CompiledTireTeacher,
        spec: TireForceNetworkSpec,
        *,
        seed: int = 1908,
        dt: float = 1.0 / 600.0,
    ) -> None:
        self.teacher = teacher
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        manifest = teacher.manifest
        topology = manifest["topology"]
        if (
            topology["circumferential_segments"] != spec.circumferential_segments
            or topology["section_segments"] != spec.section_segments
        ):
            raise ValueError("trial field must use the authoritative tire topology")
        self.rest = np.asarray(topology["rest_positions"], dtype=np.float64).reshape(
            spec.circumferential_segments, spec.section_segments, 3,
        )
        self.bead_indices = np.asarray(
            [index for ring in topology["bead_rings"] for index in ring], dtype=np.int64,
        )
        parameters = manifest["runtime_parameters"]
        self.faces = np.asarray(topology["faces"], dtype=np.int64)
        self.stiffness = float(parameters["bead_stiffness_n_per_m"])
        self.damping = float(parameters["bead_damping_n_s_per_m"])
        self.friction = float(parameters["friction_coefficient"])
        self.reference_pressure = float(parameters["reference_pressure_pa"])
        self.reference_volume = float(parameters["reference_volume_m3"])
        self.polytropic_exponent = float(parameters["gas_polytropic_exponent"])
        self.minimum_volume_fraction = float(parameters["minimum_volume_fraction"])
        self.reference_temperature = float(parameters["reference_temperature_k"])
        self.radius = float(np.max(np.linalg.norm(self.rest[:, :, :2], axis=-1)))
        bead_rest = self.rest.reshape(-1, 3)[self.bead_indices]
        # A maps individual bead forces to the exact net force/moment wrench.
        matrix = np.zeros((6, 3 * len(bead_rest)), dtype=np.float64)
        for index, position in enumerate(bead_rest):
            matrix[:3, 3 * index:3 * index + 3] = np.eye(3)
            x, y, z = position
            matrix[3:, 3 * index:3 * index + 3] = np.asarray([
                [0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0],
            ])
        self._force_from_wrench = np.linalg.pinv(matrix)

    def _command(self) -> np.ndarray:
        return np.asarray([
            self.rng.uniform(-2500.0, 2500.0),
            self.rng.uniform(1800.0, 10500.0),
            self.rng.uniform(-2200.0, 2200.0),
            self.rng.uniform(-900.0, 900.0),
            self.rng.uniform(-900.0, 900.0),
            self.rng.uniform(-1200.0, 1200.0),
        ])

    def _gas_state(self, positions: np.ndarray) -> np.ndarray:
        triangles = positions[self.faces]
        volume = float(np.sum(np.einsum(
            "ij,ij->i", triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        )) / 6.0)
        result = self.teacher.call("balloon_tire_gas", {
            "current_volume_m3": volume,
            "gas_polytropic_exponent": self.polytropic_exponent,
            "minimum_volume_fraction": self.minimum_volume_fraction,
            "reference_pressure_pa": self.reference_pressure,
            "reference_temperature_k": self.reference_temperature,
            "reference_volume_m3": self.reference_volume,
        })
        return np.asarray([
            result["gas_pressure_pa"], result["volume_ratio"],
            result["gas_temperature_k"],
        ], dtype=np.float64)

    def batch(self) -> TireTrialBatch:
        s = self.spec
        membrane = np.zeros((s.batch_size, s.history_frames,
                             s.circumferential_segments, s.section_segments, 6))
        terrain = np.zeros((s.batch_size, s.history_frames,
                            s.circumferential_segments, s.section_segments, 10))
        vehicle = np.zeros((s.batch_size, s.vehicle_state_width), dtype=np.float64)
        targets = np.zeros((s.batch_size, len(TIRE_OPERATOR_OUTPUT_NAMES)), dtype=np.float64)
        thermodynamic_state = np.zeros((s.batch_size, 3), dtype=np.float64)
        flat_rest = self.rest.reshape(-1, 3)
        bead_rest = flat_rest[self.bead_indices]
        u_angle = np.arctan2(self.rest[:, :, 1], self.rest[:, :, 0])
        bottom = np.exp(-((self.rest[:, :, 1] + self.radius) / 0.16) ** 2)

        for batch_index in range(s.batch_size):
            command = self._command()
            phase = self.rng.uniform(-np.pi, np.pi)
            slope_x, slope_z = self.rng.uniform(-0.18, 0.18, size=2)
            previous_position = None
            for frame in range(s.history_frames):
                fraction = (frame + 1) / s.history_frames
                envelope = 0.55 + 0.45 * fraction
                live_wrench = envelope * command
                bead_force = (self._force_from_wrench @ live_wrench).reshape(-1, 3)
                bead_displacement = bead_force / self.stiffness
                compression = live_wrench[1] / 260000.0
                deformation = np.zeros_like(self.rest)
                deformation[:, :, 1] += compression * bottom
                deformation[:, :, 0] += (
                    live_wrench[0] / 1800000.0 * bottom
                    + 0.002 * np.sin(2.0 * u_angle + phase) * fraction
                )
                deformation[:, :, 2] += (
                    live_wrench[2] / 1600000.0 * bottom
                    + 0.0015 * np.cos(u_angle - phase) * fraction
                )
                position = (self.rest + deformation).reshape(-1, 3)
                position[self.bead_indices] = bead_rest + bead_displacement
                velocity = (
                    np.zeros_like(position) if previous_position is None
                    else (position - previous_position) / self.dt
                )
                previous_position = position.copy()
                membrane[batch_index, frame, :, :, :3] = position.reshape(
                    s.circumferential_segments, s.section_segments, 3,
                )
                membrane[batch_index, frame, :, :, 3:] = velocity.reshape(
                    s.circumferential_segments, s.section_segments, 3,
                )
                ground_y = -self.radius + slope_x * self.rest[:, :, 0] + slope_z * self.rest[:, :, 2]
                terrain[batch_index, frame, :, :, 0] = self.rest[:, :, 0]
                terrain[batch_index, frame, :, :, 1] = ground_y
                terrain[batch_index, frame, :, :, 2] = self.rest[:, :, 2]
                normal = np.asarray([-slope_x, 1.0, -slope_z], dtype=np.float64)
                normal /= np.linalg.norm(normal)
                terrain[batch_index, frame, :, :, 3:6] = normal
                terrain[batch_index, frame, :, :, 9] = self.friction

            current = membrane[batch_index, -1].reshape(-1, 6)
            prior = membrane[batch_index, -2].reshape(-1, 6)
            targets[batch_index, :len(HUB_WRENCH_NAMES)] = self.teacher.bead_wrench(
                positions=current[self.bead_indices, :3],
                velocities=current[self.bead_indices, 3:],
                targets=bead_rest,
                target_velocities=np.zeros_like(bead_rest),
                rim_center=np.zeros(3),
                stiffness=self.stiffness,
                damping=self.damping,
            )
            thermodynamic_state[batch_index] = self._gas_state(current[:, :3])
            targets[batch_index, len(HUB_WRENCH_NAMES):] = thermodynamic_state[batch_index]
            # Canonical global conditioning: hub motion, attitude/slopes, and
            # the commanded wheel's local slip/load lanes where available.
            if s.vehicle_state_width > 7:
                vehicle[batch_index, 7] = slope_x
            if s.vehicle_state_width > 8:
                vehicle[batch_index, 8] = slope_z
            wheel_base = 20 + 7 * batch_index
            wheel_values = np.asarray([
                command[1] / 260000.0, 0.0, command[0] / 250.0,
                command[0] / 12000.0, command[2] / 12000.0,
                command[0] / 1800000.0, command[2] / 1600000.0,
            ])
            if wheel_base < s.vehicle_state_width:
                count = min(7, s.vehicle_state_width - wheel_base)
                vehicle[batch_index, wheel_base:wheel_base + count] = wheel_values[:count]

        features = pack_tire_force_features(
            s,
            membrane_history=membrane,
            terrain_history=terrain,
            rest_skin_position=self.rest,
            tire_thermodynamic_state=thermodynamic_state,
            vehicle_state=vehicle,
            dt=self.dt,
        )
        return TireTrialBatch(features, targets, membrane, terrain, vehicle)

    def reevaluate_targets(self, batch: TireTrialBatch) -> np.ndarray:
        """Re-run the native bead authority for a previously packed rig state."""

        bead_rest = self.rest.reshape(-1, 3)[self.bead_indices]
        targets = []
        for batch_index in range(self.spec.batch_size):
            current = batch.membrane_history[batch_index, -1].reshape(-1, 6)
            targets.append(self.teacher.bead_wrench(
                positions=current[self.bead_indices, :3],
                velocities=current[self.bead_indices, 3:],
                targets=bead_rest,
                target_velocities=np.zeros_like(bead_rest),
                rim_center=np.zeros(3),
                stiffness=self.stiffness,
                damping=self.damping,
            ))
        return np.asarray(targets, dtype=np.float64)


class ExactBalloonAnchorTrialGenerator(TireTrialGenerator):
    """Curriculum trials from the complete persistent balloon/contact graph.

    The four wheels begin kinematically strapped at their hubs while terrain
    motion excites the skin.  Rotation is admitted next, then progressively
    larger hub translation.  Labels are the measured six-axis anchor wrench
    plus gas state from the exact whole-skin step, never an imposed wrench or
    reconstructed bead-only surrogate.
    """

    def __init__(self, *args, exact_substeps: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.teacher.appendage is None:
            raise RuntimeError("exact anchor curriculum requires whole appendage authority")
        self.exact_substeps = int(exact_substeps)
        if self.exact_substeps <= 0:
            raise ValueError("exact substep count must be positive")
        self.curriculum_batch = 0
        self.appendage_inputs = {
            name: index for index, name in enumerate(self.teacher.appendage["inputs"])
        }

    def _release(self) -> float:
        # Deterministic four-stage release; callers can keep generating and the
        # corpus naturally retains early clamped examples.
        return min(1.0, self.curriculum_batch / 32.0)

    def batch(self) -> TireTrialBatch:
        spec = self.spec
        release = self._release()
        self.curriculum_batch += 1
        inputs, state, outputs = self.teacher.appendage_arrays()
        inputs[self.appendage_inputs["dt"]] = self.dt / self.exact_substeps
        inputs[self.appendage_inputs["gravity_y"]] = -9.81
        commands = [self._command() for _ in range(spec.batch_size)]
        hub_base_y = self.radius + 0.001
        for wheel, command in zip(("front_left", "front_right", "rear_left", "rear_right"), commands):
            inputs[self.appendage_inputs[f"{wheel}.hub_position_y"]] = hub_base_y
            inputs[self.appendage_inputs[f"{wheel}.plane_count"]] = 1.0
            inputs[self.appendage_inputs[f"{wheel}.plane_0_normal_y"]] = 1.0
            inputs[self.appendage_inputs[f"{wheel}.hub_angular_velocity_z"]] = (
                release * command[5] / 120.0
            )
        self.teacher.initialize_appendage(inputs, state)

        membrane = np.zeros((spec.batch_size, spec.history_frames,
                             spec.circumferential_segments, spec.section_segments, 6))
        terrain = np.zeros((spec.batch_size, spec.history_frames,
                            spec.circumferential_segments, spec.section_segments, 10))
        vehicle = np.zeros((spec.batch_size, spec.vehicle_state_width), dtype=np.float64)
        stride = 6 * spec.circumferential_segments * spec.section_segments
        wheels = ("front_left", "front_right", "rear_left", "rear_right")

        for frame in range(spec.history_frames):
            fraction = (frame + 1) / spec.history_frames
            for wheel_index, (wheel, command) in enumerate(zip(wheels, commands)):
                phase = 0.7 * wheel_index + fraction
                # The floor excites every clamped trial. Translation is
                # released only after angular closure has entered the corpus.
                floor_y = 0.0035 * np.sin(phase * np.pi)
                inputs[self.appendage_inputs[f"{wheel}.plane_0_point_y"]] = floor_y
                angle = release * command[5] / 900.0 * fraction
                inputs[self.appendage_inputs[f"{wheel}.hub_angle_rad"]] = angle
                inputs[self.appendage_inputs[f"{wheel}.hub_position_x"]] = (
                    release ** 2 * command[0] / 2.5e6 * fraction
                )
                inputs[self.appendage_inputs[f"{wheel}.hub_position_y"]] = (
                    hub_base_y - release ** 2 * command[1] / 4.0e6 * fraction
                )
                inputs[self.appendage_inputs[f"{wheel}.hub_velocity_x"]] = (
                    release ** 2 * command[0] / 2.5e6 / (spec.history_frames * self.dt)
                )
                inputs[self.appendage_inputs[f"{wheel}.hub_velocity_y"]] = (
                    -release ** 2 * command[1] / 4.0e6 / (spec.history_frames * self.dt)
                )
            for _ in range(self.exact_substeps):
                self.teacher.step_appendage(inputs, state, outputs)
            for wheel_index, wheel in enumerate(wheels):
                hub = np.asarray([
                    inputs[self.appendage_inputs[f"{wheel}.hub_position_{axis}"]]
                    for axis in "xyz"
                ])
                one = state[wheel_index * stride:(wheel_index + 1) * stride].reshape(-1, 6).copy()
                one[:, :3] -= hub
                membrane[wheel_index, frame] = one.reshape(
                    spec.circumferential_segments, spec.section_segments, 6
                )
                terrain[wheel_index, frame, :, :, 1] = (
                    inputs[self.appendage_inputs[f"{wheel}.plane_0_point_y"]] - hub[1]
                )
                terrain[wheel_index, frame, :, :, 3:6] = (0.0, 1.0, 0.0)
                terrain[wheel_index, frame, :, :, 9] = self.friction
                if spec.vehicle_state_width:
                    vehicle[wheel_index, 0] = release

        # Label the state shown in the final history frame with one reproducible
        # exact future step. Periodic duty trials clone these two arrays.
        trial_input = inputs.copy()
        trial_state = state.copy()
        self.teacher.step_appendage(inputs, state, outputs)
        targets = outputs.reshape(spec.batch_size, -1)[:, :len(TIRE_OPERATOR_OUTPUT_NAMES)].copy()
        target_membrane_state = np.empty((
            spec.batch_size, spec.circumferential_segments,
            spec.section_segments, len(MEMBRANE_FIELD_NAMES),
        ), dtype=np.float64)
        for wheel_index, wheel in enumerate(wheels):
            hub = np.asarray([
                inputs[self.appendage_inputs[f"{wheel}.hub_position_{axis}"]]
                for axis in "xyz"
            ])
            one = state[wheel_index * stride:(wheel_index + 1) * stride].reshape(-1, 6).copy()
            one[:, :3] -= hub
            target_membrane_state[wheel_index] = one.reshape(
                spec.circumferential_segments, spec.section_segments,
                len(MEMBRANE_FIELD_NAMES),
            )
        thermo = targets[:, len(HUB_WRENCH_NAMES):]
        features = pack_tire_force_features(
            spec, membrane_history=membrane, terrain_history=terrain,
            rest_skin_position=self.rest, tire_thermodynamic_state=thermo,
            vehicle_state=vehicle, dt=self.dt,
        )
        return TireTrialBatch(
            features, targets, membrane, terrain, vehicle,
            teacher_input=trial_input, teacher_state=trial_state,
            target_membrane_state=target_membrane_state,
        )

    def reevaluate_targets(self, batch: TireTrialBatch) -> np.ndarray:
        if batch.teacher_input is None or batch.teacher_state is None:
            raise ValueError("batch lacks an exact appendage checkpoint")
        inputs = batch.teacher_input.copy()
        state = batch.teacher_state.copy()
        outputs = np.zeros(len(self.teacher.appendage["outputs"]), dtype=np.float64)
        self.teacher.step_appendage(inputs, state, outputs)
        return outputs.reshape(self.spec.batch_size, -1)[:, :len(TIRE_OPERATOR_OUTPUT_NAMES)]


def normalized_error(predicted: np.ndarray, reference: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((predicted - reference) / scale) ** 2)))


def tire_feature_matrix(features: np.ndarray, spec: TireForceNetworkSpec) -> np.ndarray:
    """Flatten the nonduplicated periodic core for a linear/GEMM baseline."""

    halo = spec.halo
    core = np.asarray(features, dtype=np.float64)[
        :, :, halo:halo + spec.circumferential_segments,
        halo:halo + spec.section_segments,
    ]
    return core.reshape(spec.batch_size, -1)


def fit_linear_tire_operator(
    generator: TireTrialGenerator,
    *,
    batches: int = 32,
    ridge: float = 2.0e-3,
) -> tuple[LinearTireForceOperator, list[TireTrialBatch]]:
    """Fit the cheapest honest candidate by dual-form ridge regression."""

    corpus = [generator.batch() for _ in range(int(batches))]
    x = np.concatenate([
        tire_feature_matrix(batch.features, generator.spec) for batch in corpus
    ], axis=0)
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    y = np.concatenate([
        batch.target_operator_output / generator.spec.output_scale for batch in corpus
    ], axis=0)
    gram = x @ x.T
    weights = x.T @ np.linalg.solve(
        gram + float(ridge) * np.eye(len(gram), dtype=np.float64), y,
    )
    return LinearTireForceOperator(weights, x.shape[1] - 1, float(ridge)), corpus


def fit_linear_tire_state_operator(
    corpus: list[TireTrialBatch],
    spec: TireForceNetworkSpec,
    *,
    ridge: float = 2.0e-6,
) -> LinearTireStateOperator:
    """Fit one shared local state transition over all vertices and wheels."""

    x_rows, y_rows = [], []
    scale = LinearTireStateOperator.output_scale()
    for batch in corpus:
        if batch.target_membrane_state is None:
            raise ValueError("state-transition fit requires exact next-skin labels")
        halo = spec.halo
        core = batch.features[
            :, :, halo:halo + spec.circumferential_segments,
            halo:halo + spec.section_segments,
        ]
        x_rows.append(np.moveaxis(core, 1, -1).reshape(-1, spec.input_channels))
        y_rows.append(batch.target_membrane_state.reshape(-1, 6) / scale)
    x = np.concatenate(x_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    gram = x.T @ x
    weights = np.linalg.solve(
        gram + float(ridge) * np.eye(gram.shape[0], dtype=np.float64),
        x.T @ y,
    )
    return LinearTireStateOperator(weights, spec.input_channels, float(ridge))


def write_linear_tire_state_gpu_artifact(
    operator: LinearTireStateOperator,
    spec: TireForceNetworkSpec,
    destination: str | Path,
) -> dict[str, Any]:
    """Emit the shared per-vertex state propagator as a tiled GPU GEMM."""

    from .ssa_webgpu_backend import emit_gemm_module

    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = spec.batch_size * spec.circumferential_segments * spec.section_segments
    module = emit_gemm_module(
        rows, len(MEMBRANE_FIELD_NAMES), operator.input_channels + 1,
        variant="webgpu_tiled_gemm",
    )
    if not module.complete:
        raise RuntimeError(f"linear tire state GPU shortfalls: {module.shortfalls!r}")
    shader = output / "linear_tire_state_transition.wgsl"
    weights = output / "linear_tire_state_transition_weights.npy"
    parameters = output / "linear_tire_state_transition_parameters.json"
    shader.write_text(module.source, encoding="utf-8")
    gpu_weights = operator.weights.astype(np.float32)
    np.save(weights, gpu_weights)
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    payload = {
        "schema": "turing.linear-tire-state-gpu-operator.v1",
        "input_shape": [rows, operator.input_channels + 1],
        "weight_shape": list(gpu_weights.shape),
        "output_shape": [spec.batch_size, spec.circumferential_segments,
                         spec.section_segments, len(MEMBRANE_FIELD_NAMES)],
        "output_order": list(MEMBRANE_FIELD_NAMES),
        "output_scale": operator.output_scale().tolist(),
        "shader": shader.name, "weights": weights.name,
        "weights_sha256": weight_hash,
        "resident_buffers": True,
        "host_transfers_per_step": 0,
    }
    parameters.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def write_linear_tire_gpu_artifact(
    operator: LinearTireForceOperator,
    spec: TireForceNetworkSpec,
    destination: str | Path,
) -> dict[str, Any]:
    """Emit the linear candidate as the repository's canonical tiled GPU GEMM."""

    from .ssa_webgpu_backend import emit_gemm_module

    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    augmented_features = operator.feature_count + 1
    module = emit_gemm_module(
        spec.batch_size, len(TIRE_OPERATOR_OUTPUT_NAMES), augmented_features,
        variant="webgpu_tiled_gemm",
    )
    if not module.complete:
        raise RuntimeError(f"linear tire GPU shortfalls: {module.shortfalls!r}")
    shader = output / "linear_tire_hub_wrench.wgsl"
    weights = output / "linear_tire_hub_wrench_weights.npy"
    parameters = output / "linear_tire_hub_wrench_parameters.json"
    shader.write_text(module.source, encoding="utf-8")
    gpu_weights = operator.weights.astype(np.float32)
    np.save(weights, gpu_weights)
    parameter_record = {
        "schema": "turing.linear-tire-parameters.v1",
        "dtype": "float32",
        "shape": list(gpu_weights.shape),
        "weights": gpu_weights.tolist(),
        "ridge_used_for_fit": operator.ridge,
    }
    parameter_text = json.dumps(parameter_record, separators=(",", ":"))
    parameters.write_text(parameter_text, encoding="utf-8")
    metadata = module.api.to_mapping()["metadata"]
    manifest = {
        "schema": "turing.linear-tire-gpu-operator.v1",
        "input_shape": [spec.batch_size, augmented_features],
        "weight_shape": list(operator.weights.shape),
        "output_shape": [spec.batch_size, len(TIRE_OPERATOR_OUTPUT_NAMES)],
        "shader": shader.name,
        "weights": weights.name,
        "parameters_json": parameters.name,
        "parameters_sha256": hashlib.sha256(parameter_text.encode("utf-8")).hexdigest(),
        "workgroup_size": list(module.launch_plan.workgroup_size),
        "dispatch_workgroups": list(module.launch_plan.groups),
        "backend_identity": metadata.get("backend_identities"),
        "resident_buffers": True,
        "host_transfers_per_step": 0,
    }
    path = output / "linear_tire_hub_wrench.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def train_tire_operator(
    trainer: TireForceNetworkTrainer,
    generator: TireTrialGenerator,
    *,
    steps: int,
    validation_batches: int = 4,
) -> dict[str, Any]:
    validation = [generator.batch() for _ in range(validation_batches)]
    losses = []
    for _ in range(int(steps)):
        batch = generator.batch()
        losses.append(trainer.train_batch(batch.features, batch.target_operator_output))
    errors = [normalized_error(
        trainer.predict(batch.features), batch.target_operator_output, trainer.spec.output_scale,
    ) for batch in validation]
    return {
        "training_steps": int(steps),
        "first_training_loss": float(losses[0]),
        "final_training_loss": float(losses[-1]),
        "validation_normalized_rmse": float(np.mean(errors)),
        "validation_batches": validation,
    }


def prepare_native_operator_execution(
    native: NativeTireForceOperator,
    graphs: TireForceTrainingGraphs,
    trainer: TireForceNetworkTrainer,
    batch: TireTrialBatch,
):
    if trainer.model is None:
        raise RuntimeError("operator must be trained before native execution")
    feeds: dict[int, np.ndarray] = {
        native.input_value_ids["surface_state"]: batch.features,
        native.input_value_ids["target_tire_operator_normalized"]:
            batch.target_operator_output / native.spec.output_scale,
        native.input_value_ids["tire_operator_loss_weight"]:
            np.ones((1, len(TIRE_OPERATOR_OUTPUT_NAMES))),
    }
    for name, parameter in trainer.model.named_parameters():
        feeds[graphs.parameter_value_ids[name]] = np.asarray(parameter.tolist(), dtype=np.float64)
    return prepare_artifact_execution(native.artifact, feeds)


def profile_native_operator(execution, output_value_id: int, *, iterations: int = 1000) -> dict[str, Any]:
    for _ in range(8):
        execution.run()
    start = perf_counter()
    for _ in range(int(iterations)):
        execution.run()
    elapsed = perf_counter() - start
    return {
        "iterations": int(iterations),
        "seconds": elapsed,
        "microseconds_per_four_wheel_batch": 1e6 * elapsed / iterations,
        "prediction_normalized": execution.buffers[int(output_value_id)].tolist(),
    }
