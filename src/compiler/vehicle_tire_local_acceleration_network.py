"""Hub-local learned reduction of the authoritative balloon tire appendage.

The model answers only the constitutive question "how does this tire skin
accelerate?"  It has no terrain geometry, chassis, engine, road-history, or
whole-vehicle inputs.  Exact contact runs outside this model and contributes a
per-vertex hub-local boundary force field.  Setting that field to zero is the
surface-free tire evolution.  Rim/hub wrench remains an emergent reduction of
the exact bead reactions after integration; it is deliberately not a label.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ..common.tensors.abstraction import AbstractTensor
from ..common.tensors import numpy_backend as _numpy_backend  # noqa: F401
from ..common.tensors.abstract_nn import Linear
from ..common.tensors.abstract_nn.optimizer import Adam
from ..common.tensors.autograd import GradTape, autograd
from ..common.tensors.accelerator_backends.ssa_backend import SSATensorOperations, SSATensorProgram
from .process_graph_autograd import abstract_tensor_program_to_process_graph
from .vehicle_balloon_tire import build_balloon_tire_topology


MEMBRANE_LOCAL_FIELD_NAMES = (
    "position_x_m", "position_y_m", "position_z_m",
    "velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s",
)
BOUNDARY_LOCAL_FIELD_NAMES = (
    "external_force_x_n", "external_force_y_n", "external_force_z_n",
    "contact_active",
)
TIRE_THERMODYNAMIC_NAMES = ("gas_pressure_pa", "volume_ratio", "gas_temperature_k")
HUB_LOCAL_MOTION_NAMES = (
    "linear_acceleration_x_m_s2", "linear_acceleration_y_m_s2", "linear_acceleration_z_m_s2",
    "angular_velocity_x_rad_s", "angular_velocity_y_rad_s", "angular_velocity_z_rad_s",
    "angular_acceleration_x_rad_s2", "angular_acceleration_y_rad_s2", "angular_acceleration_z_rad_s2",
)
TIRE_MATERIAL_NAMES = (
    "skin_areal_density_kg_m2", "skin_thickness_m", "young_modulus_pa",
    "poisson_ratio", "kelvin_loss_pa_s", "bending_stiffness_nm",
)
TIRE_LOCAL_ACCELERATION_NAMES = (
    "local_acceleration_x_m_s2", "local_acceleration_y_m_s2", "local_acceleration_z_m_s2",
)


@dataclass(frozen=True, slots=True)
class TireLocalNetworkSpec:
    circumferential_segments: int = 16
    section_segments: int = 8
    history_frames: int = 3
    temporal_order: int = 1
    batch_size: int = 4
    feature_derivative_frequency_hz: float = 600.0
    acceleration_scale_m_s2: float = 8_000.0

    def __post_init__(self) -> None:
        if min(self.circumferential_segments, self.section_segments,
               self.history_frames, self.batch_size) <= 0:
            raise ValueError("tire-local network dimensions must be positive")
        if not 0 <= self.temporal_order < self.history_frames:
            raise ValueError("temporal_order must be below history_frames")
        if self.feature_derivative_frequency_hz <= 0 or self.acceleration_scale_m_s2 <= 0:
            raise ValueError("tire-local normalization scales must be positive")

    @property
    def local_dynamic_width(self) -> int:
        return len(MEMBRANE_LOCAL_FIELD_NAMES) + len(BOUNDARY_LOCAL_FIELD_NAMES)

    @property
    def input_channels(self) -> int:
        return (
            (self.temporal_order + 1) * self.local_dynamic_width
            + 3 + len(TIRE_THERMODYNAMIC_NAMES)
            + len(HUB_LOCAL_MOTION_NAMES) + len(TIRE_MATERIAL_NAMES)
        )

    @property
    def vertex_count(self) -> int:
        return self.circumferential_segments * self.section_segments

    @property
    def topology(self):
        return build_balloon_tire_topology(
            major_radius_m=0.42,
            section_radius_m=0.16,
            circumferential_segments=self.circumferential_segments,
            section_segments=self.section_segments,
        )

    @property
    def edge_count(self) -> int:
        return len(self.topology.edges)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.vertex_count, self.input_channels)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.vertex_count, len(TIRE_LOCAL_ACCELERATION_NAMES))

    @property
    def output_scale(self) -> np.ndarray:
        return np.full((1, 1, 3), self.acceleration_scale_m_s2, dtype=np.float64)


def tire_signed_incidence(spec: TireLocalNetworkSpec) -> np.ndarray:
    """Return one oriented row per physical edge; ``B.T`` restores reactions."""

    incidence = np.zeros((spec.edge_count, spec.vertex_count), dtype=np.float64)
    for edge_index, (left, right) in enumerate(spec.topology.edges):
        incidence[edge_index, left] = -1.0
        incidence[edge_index, right] = 1.0
    return np.broadcast_to(
        incidence[None, :, :],
        (spec.batch_size, spec.edge_count, spec.vertex_count),
    ).copy()


def tire_local_feature_names(spec: TireLocalNetworkSpec) -> tuple[str, ...]:
    names: list[str] = []
    for order in range(spec.temporal_order + 1):
        names.extend(f"d{order}.{name}" for name in (
            *MEMBRANE_LOCAL_FIELD_NAMES, *BOUNDARY_LOCAL_FIELD_NAMES,
        ))
    names.extend(f"rest.{axis}_m" for axis in "xyz")
    names.extend(TIRE_THERMODYNAMIC_NAMES)
    names.extend(HUB_LOCAL_MOTION_NAMES)
    names.extend(TIRE_MATERIAL_NAMES)
    return tuple(names)


def tire_local_feature_scales(spec: TireLocalNetworkSpec) -> np.ndarray:
    base = np.asarray([
        1, 1, 1, 30, 30, 30, 20_000, 20_000, 20_000, 1,
    ], dtype=np.float64)
    rows = [base * spec.feature_derivative_frequency_hz ** order
            for order in range(spec.temporal_order + 1)]
    rows.extend((
        np.ones(3), np.asarray([200_000, 1, 400], dtype=np.float64),
        np.asarray([2_000, 2_000, 2_000, 500, 500, 500, 5_000, 5_000, 5_000], dtype=np.float64),
        np.asarray([50, .1, 1e9, 1, 1e7, 1e4], dtype=np.float64),
    ))
    result = np.concatenate(rows)
    if result.shape != (spec.input_channels,):
        raise AssertionError("tire-local feature ABI drifted")
    return result


def _difference(history: np.ndarray, order: int, dt: float) -> np.ndarray:
    if order == 0:
        return history[:, -1]
    out = np.zeros_like(history[:, -1], dtype=np.float64)
    for offset in range(order + 1):
        out += ((-1.0) ** offset * math.comb(order, offset)
                * history[:, -1 - offset])
    return out / dt ** order


def pack_tire_local_features(
    spec: TireLocalNetworkSpec,
    *,
    membrane_history: np.ndarray,
    boundary_force_history: np.ndarray,
    rest_skin_position: np.ndarray,
    tire_thermodynamic_state: np.ndarray,
    hub_local_motion: np.ndarray,
    material_state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Pack ``[batch,time,u,v,field]`` data in the co-rotating hub frame."""

    b, t, u, v = (spec.batch_size, spec.history_frames,
                  spec.circumferential_segments, spec.section_segments)
    if tuple(membrane_history.shape) != (b, t, u, v, 6):
        raise ValueError("membrane_history must be [batch,time,u,v,6]")
    if tuple(boundary_force_history.shape) != (b, t, u, v, 4):
        raise ValueError("boundary_force_history must be [batch,time,u,v,4]")
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    for value, shape, name in (
        (tire_thermodynamic_state, (b, 3), "tire_thermodynamic_state"),
        (hub_local_motion, (b, 9), "hub_local_motion"),
        (material_state, (b, 6), "material_state"),
    ):
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    rest = np.asarray(rest_skin_position, dtype=np.float64)
    if rest.shape == (u, v, 3):
        rest = np.broadcast_to(rest[None], (b, u, v, 3))
    if rest.shape != (b, u, v, 3):
        raise ValueError("rest_skin_position must be [u,v,3] or [batch,u,v,3]")
    dynamic = np.concatenate((membrane_history, boundary_force_history), axis=-1)
    channels = [_difference(dynamic, order, float(dt))
                for order in range(spec.temporal_order + 1)]
    channels.append(rest)
    for value in (tire_thermodynamic_state, hub_local_motion, material_state):
        channels.append(np.broadcast_to(
            np.asarray(value, dtype=np.float64)[:, None, None, :],
            (b, u, v, value.shape[1]),
        ))
    field = np.concatenate(channels, axis=-1)
    field /= tire_local_feature_scales(spec)[None, None, None, :]
    return field.reshape(b, spec.vertex_count, spec.input_channels)


def teacher_local_acceleration(
    velocity_now_local: np.ndarray, velocity_next_local: np.ndarray, dt: float,
) -> np.ndarray:
    """Derive the exact teacher label without reducing anything to a hub wrench."""

    now, nxt = (np.asarray(velocity_now_local, dtype=np.float64),
                np.asarray(velocity_next_local, dtype=np.float64))
    if now.shape != nxt.shape or now.ndim != 4 or now.shape[-1] != 3:
        raise ValueError("local velocities must share [batch,u,v,3]")
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    return ((nxt - now) / float(dt)).reshape(now.shape[0], -1, 3)


class TireLocalGraphOperator:
    """Shared linear edge/node laws over the authoritative tire skin graph."""

    def __init__(self, spec: TireLocalNetworkSpec, *, like: AbstractTensor):
        self.spec = spec
        self.edge_law = Linear(spec.input_channels, 3, like=like, bias=False)
        self.node_law = Linear(spec.input_channels, 3, like=like, bias=True)

    def named_parameters(self) -> tuple[tuple[str, AbstractTensor], ...]:
        rows = []
        for name, layer in (("edge_law", self.edge_law), ("node_law", self.node_law)):
            rows.append((f"{name}.weight", layer.W))
            if layer.b is not None:
                rows.append((f"{name}.bias", layer.b))
        return tuple(rows)

    def parameters(self) -> list[AbstractTensor]:
        return [value for _name, value in self.named_parameters()]

    def assign_parameters(self, values: Mapping[str, AbstractTensor]) -> None:
        for name, layer in (("edge_law", self.edge_law), ("node_law", self.node_law)):
            layer.W = values[f"{name}.weight"]
            if layer.b is not None:
                layer.b = values[f"{name}.bias"]

    def forward(self, field: AbstractTensor, incidence: AbstractTensor) -> AbstractTensor:
        # Each undirected physical edge is stored once.  B @ x measures the
        # endpoint difference and B.T returns its message as equal/opposite
        # reactions, so internal learned forces cannot acquire a net sum.
        edge_delta = incidence @ field
        edge_message = self.edge_law.forward(edge_delta)
        internal = incidence.transpose(1, 2) @ edge_message
        return internal + self.node_law.forward(field)


@dataclass(frozen=True, slots=True)
class TireLocalTrainingGraphs:
    spec: TireLocalNetworkSpec
    feature_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    forward_input_value_ids: Mapping[str, int]
    forward_output_value_ids: tuple[int, int]
    forward_graph: Any
    forward_function: Any
    manifest: Mapping[str, Any]


def _tensor_id(value: Any) -> int:
    return int(getattr(getattr(value, "data", value), "value").id)


def build_tire_local_training_graphs(
    spec: TireLocalNetworkSpec = TireLocalNetworkSpec(),
) -> TireLocalTrainingGraphs:
    program = SSATensorProgram("tire_local_acceleration_training")
    feature = SSATensorOperations.input(program, spec.input_shape)
    incidence = SSATensorOperations.input(
        program, (spec.batch_size, spec.edge_count, spec.vertex_count),
    )
    target = SSATensorOperations.input(program, spec.output_shape)
    loss_weight = SSATensorOperations.input(program, (1, 1, 3))
    model = TireLocalGraphOperator(spec, like=AbstractTensor.tensor([0.0]))
    parameters = {name: SSATensorOperations.input(program, tuple(value.shape))
                  for name, value in model.named_parameters()}
    model.assign_parameters(parameters)
    prediction = model.forward(feature, incidence)
    error = (prediction - target) * loss_weight
    loss = (error * error).sum() / float(np.prod(spec.output_shape))
    bindings = {"tire_local_state": feature,
                "signed_skin_incidence": incidence,
                "target_local_acceleration_normalized": target,
                "local_acceleration_loss_weight": loss_weight, **parameters}
    graph = abstract_tensor_program_to_process_graph((prediction, loss), bindings=bindings)
    return TireLocalTrainingGraphs(
        spec=spec, feature_names=tire_local_feature_names(spec),
        parameter_names=tuple(parameters),
        forward_input_value_ids={name: _tensor_id(value) for name, value in bindings.items()},
        forward_output_value_ids=(_tensor_id(prediction), _tensor_id(loss)),
        forward_graph=graph, forward_function=program.function,
        manifest={
            "schema": "turing.balloon-tire-local-acceleration.v1",
            "frame": "co-rotating-hub-local",
            "input": "membrane+gas+material+hub-inertial-state+external-boundary-force-field",
            "excluded": ["terrain-geometry", "road-history", "chassis-state", "engine-state"],
            "surface_free_mode": "all external boundary force channels are zero",
            "output": list(TIRE_LOCAL_ACCELERATION_NAMES),
            "output_shape": list(spec.output_shape),
            "hub_wrench": "emergent exact bead-reaction reduction after integration",
            "batch_axis": "independent tires or trials",
            "operator": "shared-linear-edge-law+signed-incidence-reduction+shared-linear-node-law",
            "edge_count": spec.edge_count,
            "forward": "AbstractTensor-repository-SSA",
        },
    )


class TireLocalAccelerationTrainer:
    def __init__(self, spec: TireLocalNetworkSpec = TireLocalNetworkSpec(), *, lr: float = 1e-3):
        self.spec, self.lr = spec, float(lr)
        self.model: TireLocalGraphOperator | None = None
        self.optimizer: Adam | None = None

    def _ensure(self, feature: AbstractTensor) -> None:
        if self.model is None:
            self.model = TireLocalGraphOperator(self.spec, like=feature)
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train_batch(self, features: np.ndarray, target_acceleration: np.ndarray) -> float:
        if tuple(features.shape) != self.spec.input_shape or tuple(target_acceleration.shape) != self.spec.output_shape:
            raise ValueError("features or target acceleration do not match the tire-local ABI")
        autograd.tape = GradTape()
        feature = AbstractTensor.tensor(np.asarray(features, dtype=np.float64).tolist())
        target = AbstractTensor.tensor((np.asarray(target_acceleration, dtype=np.float64)
                                        / self.spec.output_scale).tolist())
        self._ensure(feature);assert self.model is not None and self.optimizer is not None
        for parameter in self.model.parameters():
            parameter._tape = autograd.tape;autograd.tape.create_tensor_node(parameter);parameter.zero_grad()
        incidence = AbstractTensor.tensor(tire_signed_incidence(self.spec).tolist())
        prediction = self.model.forward(feature, incidence);error = prediction - target
        loss = (error * error).sum() / float(np.prod(self.spec.output_shape));loss.backward()
        updated = self.optimizer.step(self.model.parameters(), [p.grad for p in self.model.parameters()])
        self.model.assign_parameters(dict(zip((n for n, _ in self.model.named_parameters()), updated, strict=True)))
        return float(loss.item())

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("train at least one batch before prediction")
        value = self.model.forward(
            AbstractTensor.tensor(np.asarray(features, dtype=np.float64).tolist()),
            AbstractTensor.tensor(tire_signed_incidence(self.spec).tolist()),
        )
        return np.asarray(value.tolist(), dtype=np.float64) * self.spec.output_scale
