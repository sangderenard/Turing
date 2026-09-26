"""AbstractTensor tire-surface operator and its trainable graph bundle.

The network approximates one expensive boundary of the scientific vehicle
graph: complete deformed tyre skin + aligned hard-surface state + whole vehicle
state -> six-axis rim/hub wrench.  Scientific balloon-skin solves generate the
targets.  No learned value replaces mass, state, or contact authority in the
teacher; this is a batchable deployment surrogate with an explicit error ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..common.tensors.abstraction import AbstractTensor
# Register the ordinary eager backend used for initialization and the Pythonic
# trainer. Source-producing SSA tensors are still constructed explicitly.
from ..common.tensors import numpy_backend as _numpy_backend  # noqa: F401
from ..common.tensors.abstract_nn import Linear, RectConv2d, Tanh
from ..common.tensors.abstract_nn.optimizer import Adam, adam_step
from ..common.tensors.autograd import GradTape, autograd
from ..common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from .process_graph_autograd import (
    ProcessGraphBackwardProduct,
    abstract_tensor_program_to_process_graph,
    compile_process_graph_backward,
    lower_training_motion_to_repository_ssa,
)
from ..transmogrifier.graph.graph_express2 import ProcessGraph


HUB_WRENCH_NAMES = (
    "force_x_n", "force_y_n", "force_z_n",
    "moment_x_nm", "moment_y_nm", "moment_z_nm",
)
TIRE_THERMODYNAMIC_NAMES = ("gas_pressure_pa", "volume_ratio", "gas_temperature_k")
TIRE_OPERATOR_OUTPUT_NAMES = (*HUB_WRENCH_NAMES, *TIRE_THERMODYNAMIC_NAMES)
MEMBRANE_FIELD_NAMES = (
    "skin_position_x_m", "skin_position_y_m", "skin_position_z_m",
    "skin_velocity_x_m_s", "skin_velocity_y_m_s", "skin_velocity_z_m_s",
)
TERRAIN_FIELD_NAMES = (
    "terrain_position_x_m", "terrain_position_y_m", "terrain_position_z_m",
    "terrain_normal_x", "terrain_normal_y", "terrain_normal_z",
    "terrain_velocity_x_m_s", "terrain_velocity_y_m_s", "terrain_velocity_z_m_s",
    "terrain_friction",
)
CANONICAL_VEHICLE_STATE_NAMES = (
    "chassis_position_x_m", "chassis_position_y_m", "chassis_position_z_m",
    "chassis_velocity_x_m_s", "chassis_velocity_y_m_s", "chassis_velocity_z_m_s",
    "chassis_roll_rad", "chassis_pitch_rad", "chassis_yaw_rad",
    "chassis_roll_rate_rad_s", "chassis_pitch_rate_rad_s", "chassis_yaw_rate_rad_s",
    "engine_speed_rad_s", "clutch_slip_rad_s", "transfer_case_ratio",
    "throttle", "brake", "steering", "fuel_mass_kg", "battery_state_of_charge",
    *(
        f"{wheel}.{field}"
        for wheel in ("front_left", "front_right", "rear_left", "rear_right")
        for field in (
            "suspension_compression_m", "suspension_velocity_m_s",
            "wheel_speed_rad_s", "longitudinal_slip_m_s", "lateral_slip_m_s",
            "sidewall_longitudinal_deformation_m", "sidewall_lateral_deformation_m",
        )
    ),
)


@dataclass(frozen=True, slots=True)
class TireForceNetworkSpec:
    """Compile-shape choices and runtime normalization for one operator."""

    circumferential_segments: int = 16
    section_segments: int = 8
    history_frames: int = 4
    temporal_order: int = 2
    vehicle_state_width: int = 48
    hidden_channels: int = 24
    latent_width: int = 32
    convolution_layers: int = 2
    batch_size: int = 4
    force_scale_n: float = 12_000.0
    moment_scale_nm: float = 4_000.0
    pressure_scale_pa: float = 200_000.0
    temperature_scale_k: float = 400.0
    feature_derivative_frequency_hz: float = 600.0

    def __post_init__(self) -> None:
        integer_fields = (
            "circumferential_segments", "section_segments", "history_frames",
            "vehicle_state_width", "hidden_channels", "latent_width",
            "convolution_layers", "batch_size",
        )
        if any(getattr(self, name) <= 0 for name in integer_fields):
            raise ValueError("tire force network dimensions must be positive")
        if not 0 <= self.temporal_order < self.history_frames:
            raise ValueError("temporal_order must be nonnegative and below history_frames")
        if self.convolution_layers != 2:
            raise ValueError("the v1 tire operator uses exactly two 3x3 convolutions")
        if (
            self.force_scale_n <= 0 or self.moment_scale_nm <= 0
            or self.pressure_scale_pa <= 0
            or self.temperature_scale_k <= 0
            or self.feature_derivative_frequency_hz <= 0
        ):
            raise ValueError("tire operator normalization scales must be positive")

    @property
    def base_field_width(self) -> int:
        return len(MEMBRANE_FIELD_NAMES) + len(TERRAIN_FIELD_NAMES)

    @property
    def input_channels(self) -> int:
        # Every temporal order sees the complete membrane+terrain field.
        # Rest skin position fixes material identity; vehicle state is spatially
        # broadcast so the convolution sees the complete conditioning state.
        return (
            (self.temporal_order + 1) * self.base_field_width
            + 3 + len(TIRE_THERMODYNAMIC_NAMES) + self.vehicle_state_width
        )

    @property
    def halo(self) -> int:
        return self.convolution_layers

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (
            self.batch_size,
            self.input_channels,
            self.circumferential_segments + 2 * self.halo,
            self.section_segments + 2 * self.halo,
        )

    @property
    def output_scale(self) -> np.ndarray:
        return np.asarray(
            [self.force_scale_n] * 3 + [self.moment_scale_nm] * 3
            + [self.pressure_scale_pa, 1.0, self.temperature_scale_k],
            # Temperature is a state/consistency output, not a force lane.
            dtype=np.float64,
        )

    @property
    def hub_wrench_scale(self) -> np.ndarray:
        return self.output_scale[:len(HUB_WRENCH_NAMES)]


def tire_force_feature_names(spec: TireForceNetworkSpec) -> tuple[str, ...]:
    names = []
    for order in range(spec.temporal_order + 1):
        names.extend(f"d{order}.{name}" for name in (
            *MEMBRANE_FIELD_NAMES, *TERRAIN_FIELD_NAMES,
        ))
    names.extend(f"rest.{axis}_m" for axis in "xyz")
    names.extend(TIRE_THERMODYNAMIC_NAMES)
    names.extend(
        CANONICAL_VEHICLE_STATE_NAMES[index]
        if index < len(CANONICAL_VEHICLE_STATE_NAMES)
        else f"vehicle_state_extension_{index}"
        for index in range(spec.vehicle_state_width)
    )
    return tuple(names)


def tire_force_feature_scales(spec: TireForceNetworkSpec) -> np.ndarray:
    """Physical scale for every named input channel, in the same ABI order."""

    base = np.asarray([
        1.0, 1.0, 1.0, 30.0, 30.0, 30.0,  # membrane x and v
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,     # terrain point and normal
        30.0, 30.0, 30.0, 2.0,             # terrain velocity and friction
    ], dtype=np.float64)
    values = [
        base * spec.feature_derivative_frequency_hz ** order
        for order in range(spec.temporal_order + 1)
    ]
    values.append(np.ones(3, dtype=np.float64))
    values.append(np.asarray(
        [spec.pressure_scale_pa, 1.0, spec.temperature_scale_k], dtype=np.float64,
    ))
    vehicle = np.ones(spec.vehicle_state_width, dtype=np.float64)
    for index, name in enumerate(CANONICAL_VEHICLE_STATE_NAMES[:spec.vehicle_state_width]):
        if "position" in name or "deformation" in name or "compression" in name:
            vehicle[index] = 2.0
        elif "velocity" in name or "slip" in name:
            vehicle[index] = 50.0
        elif "rate" in name:
            vehicle[index] = 25.0
        elif "roll_rad" in name or "pitch_rad" in name or "yaw_rad" in name:
            vehicle[index] = np.pi
        elif "engine_speed" in name or "clutch_slip" in name or "wheel_speed" in name:
            vehicle[index] = 1500.0
        elif "fuel_mass" in name:
            vehicle[index] = 150.0
    values.append(vehicle)
    scales = np.concatenate(values)
    if scales.shape != (spec.input_channels,):
        raise AssertionError("tire feature scale ABI drifted from feature names")
    return scales


def _backward_difference(history: np.ndarray, order: int, dt: float) -> np.ndarray:
    """Latest backward finite difference, preserving every spatial field."""

    if order == 0:
        return history[:, -1]
    result = np.zeros_like(history[:, -1], dtype=np.float64)
    for offset in range(order + 1):
        coefficient = (-1.0) ** offset * math.comb(order, offset)
        result += coefficient * history[:, -1 - offset]
    return result / float(dt) ** order


def pack_tire_force_features(
    spec: TireForceNetworkSpec,
    *,
    membrane_history: np.ndarray,
    terrain_history: np.ndarray,
    rest_skin_position: np.ndarray,
    tire_thermodynamic_state: np.ndarray,
    vehicle_state: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Pack physical histories into the operator's periodic channel field.

    History arrays are ``[batch,time,u,v,field]``. Terrain positions and skin
    positions must already share the hub-local frame. ``vehicle_state`` is
    ``[batch,G]`` and is broadcast over the tyre field. The returned wrapped
    halo makes both 3x3 convolutions periodic without asking a backend to
    invent a circular-padding operation.
    """

    b, t, u, v = (
        spec.batch_size, spec.history_frames,
        spec.circumferential_segments, spec.section_segments,
    )
    expected_membrane = (b, t, u, v, len(MEMBRANE_FIELD_NAMES))
    expected_terrain = (b, t, u, v, len(TERRAIN_FIELD_NAMES))
    if tuple(membrane_history.shape) != expected_membrane:
        raise ValueError(f"membrane_history must have shape {expected_membrane}")
    if tuple(terrain_history.shape) != expected_terrain:
        raise ValueError(f"terrain_history must have shape {expected_terrain}")
    if tuple(rest_skin_position.shape) not in {(u, v, 3), (b, u, v, 3)}:
        raise ValueError("rest_skin_position must be [u,v,3] or [batch,u,v,3]")
    if tuple(vehicle_state.shape) != (b, spec.vehicle_state_width):
        raise ValueError(
            f"vehicle_state must have shape {(b, spec.vehicle_state_width)}"
        )
    if tuple(tire_thermodynamic_state.shape) != (b, len(TIRE_THERMODYNAMIC_NAMES)):
        raise ValueError(
            "tire_thermodynamic_state must contain pressure, volume ratio, and temperature per wheel"
        )
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("feature sample dt must be finite and positive")

    history = np.concatenate((membrane_history, terrain_history), axis=-1)
    channels = [
        np.moveaxis(_backward_difference(history, order, dt), -1, 1)
        for order in range(spec.temporal_order + 1)
    ]
    rest = np.asarray(rest_skin_position, dtype=np.float64)
    if rest.ndim == 3:
        rest = np.broadcast_to(rest[None, ...], (b, u, v, 3))
    channels.append(np.moveaxis(rest, -1, 1))
    channels.append(np.broadcast_to(
        np.asarray(tire_thermodynamic_state, dtype=np.float64)[:, :, None, None],
        (b, len(TIRE_THERMODYNAMIC_NAMES), u, v),
    ))
    channels.append(np.broadcast_to(
        vehicle_state[:, :, None, None], (b, spec.vehicle_state_width, u, v)
    ))
    field = np.concatenate(channels, axis=1).astype(np.float64, copy=False)
    field = field / tire_force_feature_scales(spec)[None, :, None, None]
    return np.pad(
        field,
        ((0, 0), (0, 0), (spec.halo, spec.halo), (spec.halo, spec.halo)),
        mode="wrap",
    )


def reduce_teacher_bead_wrenches(
    bead_rim_forces: np.ndarray,
    bead_rim_moments: np.ndarray,
) -> np.ndarray:
    """Reduce the scientific bead/rim outputs to a six-axis training target.

    Both arrays are ``[batch,bead,xyz]`` and come directly from the compiled
    equal/opposite bead constraint. Consequently the label is the exact graph
    boundary consumed by the wheel bearing, not a force reconstructed from a
    rendered contact patch.
    """

    forces = np.asarray(bead_rim_forces, dtype=np.float64)
    moments = np.asarray(bead_rim_moments, dtype=np.float64)
    if forces.ndim != 3 or forces.shape[-1] != 3 or moments.shape != forces.shape:
        raise ValueError("bead rim force and moment arrays must share [batch,bead,3]")
    return np.concatenate((forces.sum(axis=1), moments.sum(axis=1)), axis=1)


class TireHubWrenchOperator:
    """Two periodic spatial convolutions and a compact six-wrench head."""

    def __init__(self, spec: TireForceNetworkSpec, *, like: AbstractTensor):
        self.spec = spec
        self.conv0 = RectConv2d(
            spec.input_channels, spec.hidden_channels, 3, padding=0,
            like=like,
        )
        self.conv1 = RectConv2d(
            spec.hidden_channels, spec.hidden_channels, 3, padding=0,
            like=like,
        )
        self.dense0 = Linear(
            spec.hidden_channels, spec.latent_width, like=like,
            _label_prefix="tire_operator.dense0",
        )
        self.dense1 = Linear(
            spec.latent_width, len(TIRE_OPERATOR_OUTPUT_NAMES), like=like,
            _label_prefix="tire_operator.hub_wrench",
        )
        self.activation = Tanh()

    def named_parameters(self) -> tuple[tuple[str, AbstractTensor], ...]:
        rows = []
        for prefix, layer in (
            ("conv0", self.conv0), ("conv1", self.conv1),
            ("dense0", self.dense0), ("dense1", self.dense1),
        ):
            rows.append((f"{prefix}.weight", layer.W))
            if layer.b is not None:
                rows.append((f"{prefix}.bias", layer.b))
        return tuple(rows)

    def parameters(self) -> list[AbstractTensor]:
        return [value for _name, value in self.named_parameters()]

    def assign_parameters(self, values: Mapping[str, AbstractTensor]) -> None:
        for prefix, layer in (
            ("conv0", self.conv0), ("conv1", self.conv1),
            ("dense0", self.dense0), ("dense1", self.dense1),
        ):
            layer.W = values[f"{prefix}.weight"]
            if layer.b is not None:
                layer.b = values[f"{prefix}.bias"]

    def forward(self, field: AbstractTensor) -> AbstractTensor:
        hidden = self.activation.forward(self.conv0.forward(field))
        hidden = self.activation.forward(self.conv1.forward(hidden))
        # Both valid 3x3 convolutions consume the wrapped two-cell halo and
        # recover exactly the authored UxV skin field here.
        pooled = hidden.sum(dim=3).sum(dim=2) / float(
            self.spec.circumferential_segments * self.spec.section_segments
        )
        latent = self.activation.forward(self.dense0.forward(pooled))
        return self.dense1.forward(latent)


@dataclass(frozen=True, slots=True)
class TireForceTrainingGraphs:
    spec: TireForceNetworkSpec
    feature_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_value_ids: Mapping[str, int]
    forward_input_value_ids: Mapping[str, int]
    forward_output_value_ids: tuple[int, int]
    forward_graph: ProcessGraph
    backward: ProcessGraphBackwardProduct | None
    backward_ssa: Any | None
    adam_graph: ProcessGraph
    forward_function: Any
    adam_function: Any
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NativeTireForceOperator:
    """The repository-SSA forward graph compiled as a native shared library."""

    artifact: Any
    input_value_ids: Mapping[str, int]
    output_value_ids: tuple[int, int]
    spec: TireForceNetworkSpec


def compile_tire_force_forward_native(
    graphs: TireForceTrainingGraphs,
    destination: str | Path,
) -> NativeTireForceOperator:
    """Lower the exact forward graph to native LLVM without a second model."""

    from ..transmogrifier.ssa import IRModule, Instr
    from ..transmogrifier.ssa_registry import Handler
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    function = copy.deepcopy(graphs.forward_function)
    values = {int(value.id): value for value in function.args}
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                values[int(instruction.res.id)] = instruction.res
            for value in instruction.args:
                values.setdefault(int(value.id), value)
    missing = [value_id for value_id in graphs.forward_output_value_ids if value_id not in values]
    if missing:
        raise RuntimeError(f"tire forward outputs are absent from SSA: {missing!r}")
    entry = function.blocks.get("entry") or next(iter(function.blocks.values()))
    if not any(instruction.op == Handler.Ret.value for instruction in entry.instrs):
        entry.instrs.append(Instr(
            Handler.Ret.value,
            [values[value_id] for value_id in graphs.forward_output_value_ids],
            None,
        ))
    function.metadata["named_outputs"] = (
        ("hub_wrench_normalized", graphs.forward_output_value_ids[0]),
        ("weighted_loss", graphs.forward_output_value_ids[1]),
    )
    emitted = emit_ssa_function_to_llvm(
        IRModule({function.name: function}), function.name,
        entry_name="tire_hub_wrench_operator",
    )
    if emitted.shortfalls:
        raise RuntimeError(f"tire forward LLVM shortfalls: {emitted.shortfalls!r}")
    artifact = compile_artifact(emitted, directory=Path(destination).resolve())
    return NativeTireForceOperator(
        artifact=artifact,
        input_value_ids=dict(graphs.forward_input_value_ids),
        output_value_ids=graphs.forward_output_value_ids,
        spec=graphs.spec,
    )


def _tensor_id(value: Any) -> int:
    return int(getattr(getattr(value, "data", value), "value").id)


def build_tire_force_training_graphs(
    spec: TireForceNetworkSpec = TireForceNetworkSpec(),
    *,
    include_static_backward: bool = False,
) -> TireForceTrainingGraphs:
    """Materialize forward and Adam graphs, with an optional static reverse.

    The default does not persist a reverse graph. Symbolic teacher derivatives
    and compiled network-layer derivatives feed the named gradient slots of
    Adam directly. ``include_static_backward`` remains an audit/debug option,
    not a training dependency.
    """

    program = SSATensorProgram("tire_hub_wrench_training")
    feature = SSATensorOperations.input(program, spec.input_shape)
    target = SSATensorOperations.input(program, (spec.batch_size, len(TIRE_OPERATOR_OUTPUT_NAMES)))
    loss_weight = SSATensorOperations.input(program, (1, len(TIRE_OPERATOR_OUTPUT_NAMES)))
    # Parameter initialization is intentionally outside the SSA program: the
    # program receives every weight as a live input immediately below. Some
    # AbstractNN constructors allocate eager gradient buffers while creating
    # their initial values, which is not a numerical operation the SSA source
    # backend should be asked to execute.
    initialization_like = AbstractTensor.tensor([0.0])
    model = TireHubWrenchOperator(spec, like=initialization_like)
    parameter_inputs = {
        name: SSATensorOperations.input(program, tuple(value.shape))
        for name, value in model.named_parameters()
    }
    model.assign_parameters(parameter_inputs)
    prediction = model.forward(feature)
    weighted_error = (prediction - target) * loss_weight
    loss = (weighted_error * weighted_error).sum() / float(
        spec.batch_size * len(TIRE_OPERATOR_OUTPUT_NAMES)
    )
    bindings = {
        "surface_state": feature,
        "target_tire_operator_normalized": target,
        "tire_operator_loss_weight": loss_weight,
        **parameter_inputs,
    }
    forward_graph = abstract_tensor_program_to_process_graph(
        (prediction, loss), bindings=bindings,
    )
    parameter_ids = tuple(_tensor_id(parameter_inputs[name]) for name in parameter_inputs)
    backward = (
        compile_process_graph_backward(
            forward_graph,
            outputs=(int(forward_graph.roots[1]),),
            wrt=parameter_ids,
            packaging="independent",
            unit_loss_seed=True,
        )
        if include_static_backward else None
    )
    backward_ssa = None

    adam_program = SSATensorProgram("tire_hub_wrench_adam")
    adam_bindings: dict[str, Any] = {}
    updated = []
    step = SSATensorOperations.input(adam_program, ())
    learning_rate = SSATensorOperations.input(adam_program, ())
    beta1 = SSATensorOperations.input(adam_program, ())
    beta2 = SSATensorOperations.input(adam_program, ())
    epsilon = SSATensorOperations.input(adam_program, ())
    beta1_power = SSATensorOperations.input(adam_program, ())
    beta2_power = SSATensorOperations.input(adam_program, ())
    adam_bindings.update({
        "adam_step": step, "adam_learning_rate": learning_rate,
        "adam_beta1": beta1, "adam_beta2": beta2, "adam_epsilon": epsilon,
        "adam_beta1_power_next": beta1_power,
        "adam_beta2_power_next": beta2_power,
    })
    for name, parameter in parameter_inputs.items():
        shape = tuple(parameter.shape)
        p = SSATensorOperations.input(adam_program, shape)
        g = SSATensorOperations.input(adam_program, shape)
        m = SSATensorOperations.input(adam_program, shape)
        v = SSATensorOperations.input(adam_program, shape)
        safe = name.replace(".", "__")
        adam_bindings.update({
            f"parameter__{safe}": p, f"gradient__{safe}": g,
            f"moment1__{safe}": m, f"moment2__{safe}": v,
        })
        p_new, m_new, v_new, t_new = adam_step(
            p, g, m, v, step,
            lr=learning_rate, beta1=beta1, beta2=beta2, eps=epsilon,
            beta1_power=beta1_power, beta2_power=beta2_power,
        )
        updated.extend((p_new, m_new, v_new))
    updated.append(t_new)
    adam_graph = abstract_tensor_program_to_process_graph(
        tuple(updated), bindings=adam_bindings,
    )
    return TireForceTrainingGraphs(
        spec=spec,
        feature_names=tire_force_feature_names(spec),
        parameter_names=tuple(parameter_inputs),
        parameter_value_ids={name: _tensor_id(value) for name, value in parameter_inputs.items()},
        forward_input_value_ids={
            name: _tensor_id(value) for name, value in bindings.items()
        },
        forward_output_value_ids=(
            _tensor_id(prediction), _tensor_id(loss),
        ),
        forward_graph=forward_graph,
        backward=backward,
        backward_ssa=backward_ssa,
        adam_graph=adam_graph,
        forward_function=program.function,
        adam_function=adam_program.function,
        manifest={
            "schema": "turing.balloon-tire-hub-wrench-operator.v1",
            "teacher": "compiled-balloon-skin-soft-hard-contact-graph",
            "input": "full-periodic-membrane-and-terrain-field-plus-vehicle-state",
            "temporal_features": "backward-finite-difference-orders",
            "output": list(TIRE_OPERATOR_OUTPUT_NAMES),
            "output_scale": spec.output_scale.tolist(),
            "batch_axis": "independent-wheel-samples-four-live-wheels-or-training-corpus",
            "forward": "AbstractTensor-repository-SSA",
            "gradient_authority": (
                "optional-static-ProcessGraph-adjoint"
                if include_static_backward
                else "compiled-symbolic-and-AbstractNN-layer-derivatives"
            ),
            "stored_backward_graph_required": False,
            "optimizer": "functional-AbstractTensor-Adam-with-runtime-state",
            "deployment_rule": "surrogate-must-pass-teacher-error-and-passivity-gates",
        },
    )


class TireForceNetworkTrainer:
    """Pythonic eager trainer using the same AbstractNN model and Adam law."""

    def __init__(self, spec: TireForceNetworkSpec = TireForceNetworkSpec(), *, lr: float = 1e-3):
        self.spec = spec
        self.lr = float(lr)
        self.model: TireHubWrenchOperator | None = None
        self.optimizer: Adam | None = None

    def _ensure_model(self, feature: AbstractTensor) -> None:
        if self.model is None:
            self.model = TireHubWrenchOperator(self.spec, like=feature)
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train_batch(self, features: np.ndarray, target_hub_wrench: np.ndarray) -> float:
        if tuple(features.shape) != self.spec.input_shape:
            raise ValueError(f"features must have shape {self.spec.input_shape}")
        expected_target = (self.spec.batch_size, len(TIRE_OPERATOR_OUTPUT_NAMES))
        if tuple(target_hub_wrench.shape) != expected_target:
            raise ValueError(f"target_hub_wrench must have shape {expected_target}")
        autograd.tape = GradTape()
        feature = AbstractTensor.tensor(np.asarray(features, dtype=np.float64).tolist())
        target = AbstractTensor.tensor(
            (np.asarray(target_hub_wrench, dtype=np.float64) / self.spec.output_scale).tolist()
        )
        self._ensure_model(feature)
        assert self.model is not None and self.optimizer is not None
        for parameter in self.model.parameters():
            parameter._tape = autograd.tape
            autograd.tape.create_tensor_node(parameter)
            parameter.zero_grad()
        prediction = self.model.forward(feature)
        error = prediction - target
        loss = (error * error).sum() / float(
            self.spec.batch_size * len(TIRE_OPERATOR_OUTPUT_NAMES)
        )
        loss.backward()
        parameters = self.model.parameters()
        gradients = [parameter.grad for parameter in parameters]
        updated = self.optimizer.step(parameters, gradients)
        self.model.assign_parameters(dict(zip(
            (name for name, _value in self.model.named_parameters()), updated, strict=True,
        )))
        return float(loss.item())

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("train at least one batch before prediction")
        feature = AbstractTensor.tensor(np.asarray(features, dtype=np.float64).tolist())
        normalized = np.asarray(self.model.forward(feature).tolist(), dtype=np.float64)
        return normalized * self.spec.output_scale

    def create_reference_workshare(self, *, config=None):
        """Create the exact-first controller for this network's wrench scale."""

        from .vehicle_tire_force_workshare import (
            TireForceReferenceWorkShare,
            TireForceWorkShareConfig,
        )

        return TireForceReferenceWorkShare(
            output_scale=self.spec.hub_wrench_scale,
            config=config if config is not None else TireForceWorkShareConfig(),
        )
