"""LLVM compiler/runtime for a persistent perforated transition cell.

Forward and the two-output VJP are compiled from one AbstractTensor graph.
The VJP differentiates with respect to both parameters and the incoming
hidden state.  ``train_trajectories`` therefore performs true reverse-mode
learning through complete trajectories without tape autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..common.tensors.abstract_nn.perforated_recurrent import (
    PerforatedRecurrentTransition,
)
from ..common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from .llvm_training_runtime import (
    NativeGraphForward,
    NativeGraphReverse,
    compile_native_graph_forward,
    compile_native_graph_reverse,
)
from .ssa_llvm_backend import prepare_artifact_execution


PARAMETER_NAMES = (
    "input_weight", "recurrent_weight", "hidden_bias",
    "dendrite_input_weight", "dendrite_recurrent_weight",
    "dendrite_bias", "dendrite_gain", "output_weight",
    "direct_weight", "output_bias",
)


@dataclass(frozen=True)
class RecurrentStep:
    prediction: np.ndarray
    hidden: np.ndarray


@dataclass(frozen=True)
class RecurrentTrainingResult:
    initial_rollout_mse: float
    final_rollout_mse: float
    epoch_losses: tuple[float, ...]
    trajectory_count: int
    transition_count: int


@dataclass(frozen=True)
class CompiledRecurrentAdamTrajectory:
    """One LLVM entry for whole-trajectory BPTT and accumulated Adam."""

    artifact: object
    input_value_ids: Mapping[str, int]
    parameter_value_ids: Mapping[str, int]
    output_value_ids: Mapping[str, int]
    gradient_value_ids: Mapping[str, int]
    optimizer_state_value_ids: Mapping[str, object]
    trajectory_steps: int
    experience_count: int
    shapes: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class NativeRecurrentCycleResult:
    loss_bank: np.ndarray
    parameters: Mapping[str, np.ndarray]
    iteration: int
    gradient_norm: float


def run_recurrent_adam_trajectory(
    compiled: CompiledRecurrentAdamTrajectory, *, exogenous: np.ndarray,
    target_states: np.ndarray, initial_states: np.ndarray,
    state_route: np.ndarray, delta_to_state_scale: np.ndarray,
    delta_to_state_bias: np.ndarray, epochs: int, seed: int = 1729,
    learning_rate: float = 0.00035,
) -> NativeRecurrentCycleResult:
    """Execute all experiences and epochs in one call to the LLVM entry."""
    exogenous = np.asarray(exogenous, dtype=np.float64)
    target_states = np.asarray(target_states, dtype=np.float64)
    initial_states = np.asarray(initial_states, dtype=np.float64)
    count, length, in_dim = exogenous.shape
    if (count != compiled.experience_count
            or length != compiled.trajectory_steps):
        raise ValueError("experience bank does not match compiled trajectory contract")
    out_dim = target_states.shape[2]
    if target_states.shape[:2] != (count, length):
        raise ValueError("target state bank must align with experiences and transitions")
    ids = compiled.input_value_ids
    state = compiled.optimizer_state_value_ids
    hidden_dim = compiled.shapes["initial_hidden"][1]
    width = compiled.shapes["dendrite_bias"][1]
    branches = width // hidden_dim
    rng = np.random.default_rng(seed)
    parameters = {
        "input_weight": rng.normal(0.0, np.sqrt(1.0 / in_dim),
                                   compiled.shapes["input_weight"]),
        "recurrent_weight": rng.normal(0.0, 0.12 / np.sqrt(hidden_dim),
                                       compiled.shapes["recurrent_weight"]),
        "hidden_bias": np.zeros(compiled.shapes["hidden_bias"]),
        "dendrite_input_weight": rng.normal(
            0.0, np.sqrt(1.0 / in_dim), compiled.shapes["dendrite_input_weight"]),
        "dendrite_recurrent_weight": rng.normal(
            0.0, 0.08 / np.sqrt(hidden_dim),
            compiled.shapes["dendrite_recurrent_weight"]),
        "dendrite_bias": rng.normal(0.0, 0.03, compiled.shapes["dendrite_bias"]),
        "dendrite_gain": np.full(compiled.shapes["dendrite_gain"], 0.03 / branches),
        "output_weight": rng.normal(0.0, 1e-3, compiled.shapes["output_weight"]),
        "direct_weight": np.zeros(compiled.shapes["direct_weight"]),
        "output_bias": np.zeros(compiled.shapes["output_bias"]),
    }
    route = np.asarray([
        [1.0 if hidden_index == branch_index // branches else 0.0
         for hidden_index in range(hidden_dim)]
        for branch_index in range(width)
    ], dtype=np.float64)
    selected_steps = int(epochs) * count
    values = {
        ids["initial_state"]: np.ascontiguousarray(initial_states[:, None, :]),
        ids["initial_hidden"]: np.zeros((count, 1, hidden_dim)),
        ids["state_route"]: np.ascontiguousarray(state_route, dtype=np.float64),
        ids["delta_to_state_scale"]: np.asarray(delta_to_state_scale).reshape(1, out_dim),
        ids["delta_to_state_bias"]: np.asarray(delta_to_state_bias).reshape(1, out_dim),
        ids["loss_scale"]: np.full((1, out_dim), 1.0 / (length * out_dim)),
        ids["dendrite_route"]: route,
        ids["dendrite_mask"]: np.ones((1, width)),
        state["steps"]: np.asarray(selected_steps, dtype=np.int32),
        state["learning_rate"]: np.asarray(float(learning_rate)),
        state["beta1"]: np.asarray(0.9), state["beta2"]: np.asarray(0.999),
        state["epsilon"]: np.asarray(1e-8),
        state["beta1_power"]: np.asarray(1.0),
        state["beta2_power"]: np.asarray(1.0),
        state["iteration"]: np.asarray(0, dtype=np.int32),
        state["gradient_norm"]: np.asarray(0.0),
        state["clipped_gradient_norm"]: np.asarray(0.0),
    }
    for step_index in range(length):
        values[ids[f"exogenous_{step_index}"]] = np.ascontiguousarray(
            exogenous[:, step_index:step_index + 1, :])
        values[ids[f"target_state_{step_index}"]] = np.ascontiguousarray(
            target_states[:, step_index:step_index + 1, :])
    for name, parameter in parameters.items():
        parameter_id = compiled.parameter_value_ids[name]
        values[parameter_id] = parameter
        values[state["first_moment"][parameter_id]] = np.zeros_like(parameter)
        values[state["second_moment"][parameter_id]] = np.zeros_like(parameter)
        values[state["gradient_accumulator"][parameter_id]] = np.zeros_like(parameter)
    execution = prepare_artifact_execution(compiled.artifact, values).run()
    learned = {
        name: execution.buffers[compiled.parameter_value_ids[name]].copy()
        for name in PARAMETER_NAMES
    }
    return NativeRecurrentCycleResult(
        execution.buffers[compiled.output_value_ids["loss_0"]].reshape(-1).copy(),
        learned, int(execution.buffers[state["iteration"]]),
        float(execution.buffers[state["clipped_gradient_norm"]]))


def _build_graph(in_dim: int, hidden_dim: int, out_dim: int, branches: int):
    program = SSATensorProgram("perforated_recurrent_transition")
    width = hidden_dim * branches
    shapes = {
        "x": (1, in_dim), "hidden": (1, hidden_dim),
        "input_weight": (in_dim, hidden_dim),
        "recurrent_weight": (hidden_dim, hidden_dim),
        "hidden_bias": (1, hidden_dim),
        "dendrite_input_weight": (in_dim, width),
        "dendrite_recurrent_weight": (hidden_dim, width),
        "dendrite_bias": (1, width), "dendrite_gain": (1, width),
        "dendrite_route": (width, hidden_dim),
        "dendrite_mask": (1, width),
        "output_weight": (hidden_dim, out_dim),
        "direct_weight": (in_dim, out_dim), "output_bias": (1, out_dim),
    }
    bindings = {name: SSATensorOperations.input(program, shape)
                for name, shape in shapes.items()}
    cell = PerforatedRecurrentTransition()
    prediction, hidden_next = cell.forward(
        bindings["x"], bindings["hidden"],
        **{name: bindings[name] for name in PARAMETER_NAMES},
        dendrite_route=bindings["dendrite_route"],
        dendrite_mask=bindings["dendrite_mask"],
    )
    return (prediction, hidden_next), bindings, shapes


def compile_recurrent_adam_trajectory(
    directory: str | Path, *, in_dim: int, hidden_dim: int, out_dim: int,
    trajectory_steps: int, experience_count: int,
    dendrites_per_hidden: int = 2, max_global_gradient_norm: float = 0.25,
    name: str = "engine_recurrent_trajectory_adam",
) -> CompiledRecurrentAdamTrajectory:
    """Compile recurrence, full reverse pass, accumulation, and Adam together."""
    if min(in_dim, hidden_dim, out_dim, trajectory_steps,
           experience_count, dendrites_per_hidden) < 1:
        raise ValueError("all recurrent trajectory dimensions must be positive")
    from .process_graph_autograd import (
        lower_training_motion_to_repository_ssa,
        obtain_graph_reverse,
    )
    from .ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        with_native_adam_loop,
    )

    program = SSATensorProgram("perforated_recurrent_full_trajectory")
    width = hidden_dim * dendrites_per_hidden
    shapes = {
        "initial_state": (1, out_dim), "initial_hidden": (1, hidden_dim),
        "state_route": (out_dim, in_dim),
        "delta_to_state_scale": (1, out_dim),
        "delta_to_state_bias": (1, out_dim),
        "loss_scale": (1, out_dim),
        "input_weight": (in_dim, hidden_dim),
        "recurrent_weight": (hidden_dim, hidden_dim),
        "hidden_bias": (1, hidden_dim),
        "dendrite_input_weight": (in_dim, width),
        "dendrite_recurrent_weight": (hidden_dim, width),
        "dendrite_bias": (1, width), "dendrite_gain": (1, width),
        "dendrite_route": (width, hidden_dim),
        "dendrite_mask": (1, width),
        "output_weight": (hidden_dim, out_dim),
        "direct_weight": (in_dim, out_dim), "output_bias": (1, out_dim),
    }
    bindings = {key: SSATensorOperations.input(program, shape)
                for key, shape in shapes.items()}
    for step in range(trajectory_steps):
        for prefix, shape in (("exogenous", (1, in_dim)),
                              ("target_state", (1, out_dim))):
            key = f"{prefix}_{step}"
            shapes[key] = shape
            bindings[key] = SSATensorOperations.input(program, shape)
    cell = PerforatedRecurrentTransition()
    state = bindings["initial_state"]
    hidden = bindings["initial_hidden"]
    loss = None
    prediction = None
    for step in range(trajectory_steps):
        x = bindings[f"exogenous_{step}"] + state @ bindings["state_route"]
        prediction, hidden = cell.forward(
            x, hidden, **{key: bindings[key] for key in PARAMETER_NAMES},
            dendrite_route=bindings["dendrite_route"],
            dendrite_mask=bindings["dendrite_mask"])
        state = (state + prediction * bindings["delta_to_state_scale"]
                 + bindings["delta_to_state_bias"])
        error = state - bindings[f"target_state_{step}"]
        step_loss = (error * error * bindings["loss_scale"]).sum()
        loss = step_loss if loss is None else loss + step_loss
    ids = {key: int(value.data.value.id) for key, value in bindings.items()}
    parameter_ids = tuple(ids[key] for key in PARAMETER_NAMES)
    product = obtain_graph_reverse(
        loss, bindings=bindings, wrt=parameter_ids,
        packaging="combined", unit_output_seed=True)
    if product.motion is None:
        raise RuntimeError("recurrent trajectory produced no reverse motion")
    lowering = lower_training_motion_to_repository_ssa(
        product.motion, function_name=f"{name}__motion",
        observed_outputs={"final_state": int(state.data.value.id)})
    if lowering.shortfalls:
        raise RuntimeError(f"recurrent trajectory SSA shortfalls: {lowering.shortfalls!r}")
    emitted = emit_ssa_function_to_llvm(
        lowering.module, lowering.function_name,
        entry_name=lowering.function_name)
    if emitted.shortfalls:
        raise RuntimeError(f"recurrent trajectory LLVM shortfalls: {emitted.shortfalls!r}")
    gradients = {
        key: int(product.motion.gradient_value_ids[ids[key]])
        for key in PARAMETER_NAMES
    }
    outputs = dict(lowering.outputs)
    cycled_names = (
        "initial_state", "initial_hidden",
        *(f"exogenous_{step}" for step in range(trajectory_steps)),
        *(f"target_state_{step}" for step in range(trajectory_steps)),
    )
    wrapped = with_native_adam_loop(
        emitted,
        parameter_gradient_pairs=tuple(
            (ids[key], gradients[key]) for key in PARAMETER_NAMES),
        cycled_value_ids=tuple(ids[key] for key in cycled_names)
        + (int(outputs["loss_0"]),),
        cycle_length=experience_count,
        gradient_accumulation_steps=experience_count,
        max_global_gradient_norm=max_global_gradient_norm,
        entry_name=name)
    artifact = compile_artifact(wrapped, directory=Path(directory) / "native")
    return CompiledRecurrentAdamTrajectory(
        artifact=artifact, input_value_ids=ids,
        parameter_value_ids={key: ids[key] for key in PARAMETER_NAMES},
        output_value_ids=outputs, gradient_value_ids=gradients,
        optimizer_state_value_ids=dict(artifact.optimizer_state_value_ids or {}),
        trajectory_steps=trajectory_steps, experience_count=experience_count,
        shapes=shapes)


class CompiledPerforatedRecurrent:
    """Compiled recurrent cell plus host-side trajectory optimizer.

    The host schedules complete trajectories. All numerical forward and VJP
    motions execute in LLVM; Python only carries buffers across transitions
    and applies one accumulated Adam update per trajectory.
    """

    def __init__(self, forward: NativeGraphForward, reverse: NativeGraphReverse,
                 shapes: Mapping[str, tuple[int, ...]], *, seed: int = 1729,
                 learning_rate: float = 0.002, max_gradient_norm: float = 1.0):
        self.forward_artifact = forward
        self.reverse_artifact = reverse
        self.shapes = dict(shapes)
        self.ids = dict(forward.input_value_ids)
        self.output_ids = tuple(forward.output_value_ids)
        self.learning_rate = float(learning_rate)
        self.max_gradient_norm = float(max_gradient_norm)
        self.iteration = 0
        rng = np.random.default_rng(seed)
        in_dim, hidden_dim = self.shapes["input_weight"]
        out_dim = self.shapes["output_bias"][1]
        width = self.shapes["dendrite_bias"][1]
        branches = width // hidden_dim
        self.parameters = {
            "input_weight": rng.normal(0.0, np.sqrt(1.0 / in_dim),
                                       self.shapes["input_weight"]),
            "recurrent_weight": rng.normal(0.0, 0.12 / np.sqrt(hidden_dim),
                                           self.shapes["recurrent_weight"]),
            "hidden_bias": np.zeros(self.shapes["hidden_bias"]),
            "dendrite_input_weight": rng.normal(
                0.0, np.sqrt(1.0 / in_dim), self.shapes["dendrite_input_weight"]),
            "dendrite_recurrent_weight": rng.normal(
                0.0, 0.08 / np.sqrt(hidden_dim),
                self.shapes["dendrite_recurrent_weight"]),
            "dendrite_bias": rng.normal(0.0, 0.03, self.shapes["dendrite_bias"]),
            "dendrite_gain": np.full(self.shapes["dendrite_gain"], 0.03 / branches),
            # A state-transition model begins near the identity map. Random
            # order-one deltas compound catastrophically over a long rollout.
            "output_weight": rng.normal(0.0, 1e-3,
                                        self.shapes["output_weight"]),
            "direct_weight": np.zeros(self.shapes["direct_weight"]),
            "output_bias": np.zeros(self.shapes["output_bias"]),
        }
        self.route = np.asarray([
            [1.0 if hidden_index == branch_index // branches else 0.0
             for hidden_index in range(hidden_dim)]
            for branch_index in range(width)
        ])
        self.mask = np.ones(self.shapes["dendrite_mask"])
        self.first = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.gradient_ids = {
            name: reverse.gradient_value_ids[self.ids[name]]
            for name in (*PARAMETER_NAMES, "x", "hidden")
        }

    @classmethod
    def compile(cls, directory: str | Path, *, in_dim: int, hidden_dim: int,
                out_dim: int, dendrites_per_hidden: int = 2, seed: int = 1729,
                learning_rate: float = 0.002, max_gradient_norm: float = 1.0):
        outputs, bindings, shapes = _build_graph(
            in_dim, hidden_dim, out_dim, dendrites_per_hidden)
        directory = Path(directory)
        forward = compile_native_graph_forward(
            outputs, bindings=bindings,
            source="PerforatedRecurrentTransition.forward",
            name="engine_recurrent_forward", directory=directory / "forward")
        wrt = [bindings[name].data.value.id
               for name in (*PARAMETER_NAMES, "x", "hidden")]
        reverse = compile_native_graph_reverse(
            outputs, bindings=bindings, wrt_value_ids=wrt,
            name="engine_recurrent_vjp", directory=directory / "vjp")
        return cls(forward, reverse, shapes, seed=seed,
                   learning_rate=learning_rate,
                   max_gradient_norm=max_gradient_norm)

    def _bindings(self, x: np.ndarray, hidden: np.ndarray) -> dict[int, np.ndarray]:
        values = {self.ids["x"]: np.ascontiguousarray(x, dtype=np.float64),
                  self.ids["hidden"]: np.ascontiguousarray(hidden, dtype=np.float64),
                  self.ids["dendrite_route"]: self.route,
                  self.ids["dendrite_mask"]: self.mask}
        values.update({self.ids[name]: value for name, value in self.parameters.items()})
        return values

    def forward(self, x: np.ndarray, hidden: np.ndarray) -> RecurrentStep:
        execution = prepare_artifact_execution(
            self.forward_artifact.artifact, self._bindings(x, hidden)).run()
        return RecurrentStep(
            execution.buffers[self.output_ids[0]].copy(),
            execution.buffers[self.output_ids[1]].copy())

    def rollout(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.zeros(self.shapes["hidden"], dtype=np.float64)
        predictions = []
        for row in np.asarray(x, dtype=np.float64):
            step = self.forward(row.reshape(1, -1), hidden)
            predictions.append(step.prediction[0])
            hidden = step.hidden
        return np.asarray(predictions), hidden

    def _trajectory_gradients(self, x: np.ndarray, target: np.ndarray):
        hidden = np.zeros(self.shapes["hidden"], dtype=np.float64)
        frames = []
        predictions = []
        for row in x:
            before = hidden.copy()
            step = self.forward(row.reshape(1, -1), hidden)
            frames.append((row.reshape(1, -1), before))
            predictions.append(step.prediction[0])
            hidden = step.hidden
        predictions = np.asarray(predictions)
        error = predictions - target
        seed_scale = 2.0 / float(max(1, error.size))
        hidden_adjoint = np.zeros(self.shapes["hidden"], dtype=np.float64)
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        seed_ids = self.reverse_artifact.seed_value_ids
        for index in range(len(frames) - 1, -1, -1):
            row, hidden_before = frames[index]
            values = self._bindings(row, hidden_before)
            values[seed_ids[self.output_ids[0]]] = np.ascontiguousarray(
                error[index:index + 1] * seed_scale)
            values[seed_ids[self.output_ids[1]]] = hidden_adjoint
            reverse = prepare_artifact_execution(
                self.reverse_artifact.artifact, values).run()
            for name in PARAMETER_NAMES:
                gradients[name] += reverse.buffers[self.gradient_ids[name]]
            hidden_adjoint = reverse.buffers[self.gradient_ids["hidden"]].copy()
        return float(np.mean(error * error)), gradients

    def train_trajectory(self, x: np.ndarray, target: np.ndarray) -> float:
        loss, gradients = self._trajectory_gradients(
            np.asarray(x, dtype=np.float64), np.asarray(target, dtype=np.float64))
        self._apply_gradients(gradients)
        return loss

    def _apply_gradients(self, gradients: Mapping[str, np.ndarray]) -> None:
        norm = float(np.sqrt(sum(np.sum(value * value) for value in gradients.values())))
        if self.max_gradient_norm > 0.0 and norm > self.max_gradient_norm:
            scale = self.max_gradient_norm / max(norm, 1e-30)
            gradients = {name: value * scale for name, value in gradients.items()}
        self.iteration += 1
        beta1, beta2 = 0.9, 0.999
        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            self.first[name] = beta1 * self.first[name] + (1.0 - beta1) * gradient
            self.second[name] = beta2 * self.second[name] + (1.0 - beta2) * gradient * gradient
            first = self.first[name] / (1.0 - beta1 ** self.iteration)
            second = self.second[name] / (1.0 - beta2 ** self.iteration)
            parameter -= self.learning_rate * first / (np.sqrt(second) + 1e-8)

    def rollout_closed_loop(
        self, x: np.ndarray, initial_state: np.ndarray,
        *, state_columns: np.ndarray, delta_to_state_scale: np.ndarray,
        delta_to_state_bias: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run while the network, rather than teacher rows, owns machine state."""
        hidden = np.zeros(self.shapes["hidden"], dtype=np.float64)
        state = np.asarray(initial_state, dtype=np.float64).reshape(1, -1).copy()
        states = []
        for source in np.asarray(x, dtype=np.float64):
            row = source.reshape(1, -1).copy()
            row[:, state_columns] = state
            step = self.forward(row, hidden)
            state = (state + step.prediction * delta_to_state_scale
                     + delta_to_state_bias)
            states.append(state[0].copy())
            hidden = step.hidden
        return np.asarray(states), hidden

    def _closed_loop_gradients(
        self, x: np.ndarray, target_states: np.ndarray,
        *, initial_state: np.ndarray, state_columns: np.ndarray,
        delta_to_state_scale: np.ndarray, delta_to_state_bias: np.ndarray,
    ):
        hidden = np.zeros(self.shapes["hidden"], dtype=np.float64)
        state = np.asarray(initial_state, dtype=np.float64).reshape(1, -1).copy()
        frames, predicted_states = [], []
        for source in np.asarray(x, dtype=np.float64):
            row = source.reshape(1, -1).copy()
            row[:, state_columns] = state
            hidden_before = hidden.copy()
            step = self.forward(row, hidden)
            state = (state + step.prediction * delta_to_state_scale
                     + delta_to_state_bias)
            frames.append((row, hidden_before))
            predicted_states.append(state[0].copy())
            hidden = step.hidden
        predicted_states = np.asarray(predicted_states)
        error = predicted_states - np.asarray(target_states, dtype=np.float64)
        state_adjoint = np.zeros_like(state)
        hidden_adjoint = np.zeros(self.shapes["hidden"], dtype=np.float64)
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        seed_ids = self.reverse_artifact.seed_value_ids
        loss_scale = 2.0 / float(max(1, error.size))
        for index in range(len(frames) - 1, -1, -1):
            row, hidden_before = frames[index]
            next_state_adjoint = state_adjoint + error[index:index + 1] * loss_scale
            values = self._bindings(row, hidden_before)
            values[seed_ids[self.output_ids[0]]] = np.ascontiguousarray(
                next_state_adjoint * delta_to_state_scale)
            values[seed_ids[self.output_ids[1]]] = hidden_adjoint
            reverse = prepare_artifact_execution(
                self.reverse_artifact.artifact, values).run()
            for name in PARAMETER_NAMES:
                gradients[name] += reverse.buffers[self.gradient_ids[name]]
            input_adjoint = reverse.buffers[self.gradient_ids["x"]]
            state_adjoint = next_state_adjoint + input_adjoint[:, state_columns]
            hidden_adjoint = reverse.buffers[self.gradient_ids["hidden"]].copy()
        return float(np.mean(error * error)), gradients

    def train_closed_loop_trajectory(
        self, x: np.ndarray, target_states: np.ndarray,
        *, initial_state: np.ndarray, state_columns: np.ndarray,
        delta_to_state_scale: np.ndarray, delta_to_state_bias: np.ndarray,
    ) -> float:
        loss, gradients = self._closed_loop_gradients(
            x, target_states, initial_state=initial_state,
            state_columns=state_columns,
            delta_to_state_scale=delta_to_state_scale,
            delta_to_state_bias=delta_to_state_bias)
        self._apply_gradients(gradients)
        return loss

    def train_trajectories(self, trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
                           *, epochs: int = 10, seed: int = 1729
                           ) -> RecurrentTrainingResult:
        if not trajectories:
            raise ValueError("at least one complete trajectory is required")
        rng = np.random.default_rng(seed)
        initial = float(np.mean([
            np.mean((self.rollout(x)[0] - y) ** 2) for x, y in trajectories]))
        history = []
        for _ in range(int(epochs)):
            losses = []
            for index in rng.permutation(len(trajectories)):
                x, y = trajectories[int(index)]
                losses.append(self.train_trajectory(x, y))
            history.append(float(np.mean(losses)))
        final = float(np.mean([
            np.mean((self.rollout(x)[0] - y) ** 2) for x, y in trajectories]))
        return RecurrentTrainingResult(
            initial, final, tuple(history), len(trajectories),
            sum(len(x) for x, _ in trajectories))


__all__ = [
    "CompiledPerforatedRecurrent", "RecurrentStep", "RecurrentTrainingResult",
]
