"""Adam training driven by a compiled perforated forward and VJP.

LLVM owns every network forward and parameter gradient. Python owns only the
Adam state update and minibatch scheduling; no eager tape participates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .perforated_network_llvm import (
    CompiledPerforatedAdamChunk,
    CompiledPerforatedNetwork,
    compile_perforated_adam_chunk,
    compile_perforated_network,
)
from .ssa_llvm_backend import prepare_artifact_execution


@dataclass(frozen=True)
class NativeAdamStep:
    loss: float
    prediction: np.ndarray
    gradient_norm: float
    parameter_norm: float


class CompiledPerforatedAdam:
    """Stateful Adam wrapper around the native LLVM forward/VJP contract."""

    parameter_names = (
        "base_weight", "base_bias", "dendrite_weight",
        "dendrite_bias", "dendrite_gain",
    )

    def __init__(self, compiled: CompiledPerforatedNetwork, *, seed: int = 1729,
                 learning_rate: float = 0.006, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 training_cycle: CompiledPerforatedAdamChunk | None = None) -> None:
        self.compiled = compiled
        self.training_cycle = training_cycle
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.iteration = 0
        self.beta1_power = 1.0
        self.beta2_power = 1.0
        ports = {port.name: port for port in compiled.contract.inputs}
        self.ids = {name: port.value_id for name, port in ports.items()}
        self.shapes = {name: port.shape for name, port in ports.items()}
        rng = np.random.default_rng(seed)
        in_dim, out_dim = self.shapes["base_weight"]
        width = self.shapes["dendrite_bias"][1]
        branches = compiled.contract.dendrites_per_neuron
        self.parameters: dict[str, np.ndarray] = {
            "base_weight": rng.normal(0.0, np.sqrt(2.0 / (in_dim + out_dim)),
                                      size=(in_dim, out_dim)),
            "base_bias": np.zeros((1, out_dim), dtype=np.float64),
            "dendrite_weight": rng.normal(0.0, np.sqrt(1.0 / in_dim),
                                           size=(in_dim, width)),
            "dendrite_bias": rng.normal(0.0, 0.05, size=(1, width)),
            "dendrite_gain": np.full((1, width), 0.05 / branches),
        }
        self.route = np.asarray([
            [1.0 if output == branch // branches else 0.0
             for output in range(out_dim)]
            for branch in range(width)
        ], dtype=np.float64)
        self.dendrite_mask = np.ones(self.shapes["dendrite_mask"], dtype=np.float64)
        self.network_authority = np.ones(
            self.shapes["network_authority"], dtype=np.float64)
        self.simulator_delta = np.zeros(
            self.shapes["simulator_delta"], dtype=np.float64)
        self.first_moment = {name: np.zeros_like(value)
                             for name, value in self.parameters.items()}
        self.second_moment = {name: np.zeros_like(value)
                              for name, value in self.parameters.items()}
        self.gradient_ids = {
            port.name.removeprefix("grad_"): port.value_id
            for port in compiled.contract.gradients
        }

    @classmethod
    def compile(cls, directory: str | Path, *, batch: int, in_dim: int,
                out_dim: int, dendrites_per_neuron: int = 2,
                seed: int = 1729, learning_rate: float = 0.006,
                cycle_length: int | None = None,
                gradient_accumulation_steps: int = 1,
                max_global_gradient_norm: float | None = None,
                use_cache: bool = True):
        compiled = compile_perforated_network(
            directory, batch=batch, in_dim=in_dim, out_dim=out_dim,
            dendrites_per_neuron=dendrites_per_neuron,
            name="engine_dendrite_learner",
            use_cache=use_cache,
        )
        training_cycle = None
        if cycle_length is not None:
            training_cycle = compile_perforated_adam_chunk(
                Path(directory) / "training-cycle",
                batch=batch, in_dim=in_dim, out_dim=out_dim,
                cycle_length=cycle_length,
                dendrites_per_neuron=dendrites_per_neuron,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_global_gradient_norm=max_global_gradient_norm,
                name="engine_dendrite_training_cycle",
                use_cache=use_cache,
            )
        return cls(
            compiled, seed=seed, learning_rate=learning_rate,
            training_cycle=training_cycle)

    def _bindings(self, x: np.ndarray) -> Mapping[int, np.ndarray]:
        return {
            self.ids["x"]: np.ascontiguousarray(x, dtype=np.float64),
            self.ids["base_weight"]: self.parameters["base_weight"],
            self.ids["base_bias"]: self.parameters["base_bias"],
            self.ids["dendrite_weight"]: self.parameters["dendrite_weight"],
            self.ids["dendrite_bias"]: self.parameters["dendrite_bias"],
            self.ids["dendrite_gain"]: self.parameters["dendrite_gain"],
            self.ids["dendrite_route"]: self.route,
            self.ids["dendrite_mask"]: self.dendrite_mask,
            self.ids["network_authority"]: self.network_authority,
            self.ids["simulator_delta"]: self.simulator_delta,
        }

    def set_branch_authority(self, mask: np.ndarray) -> None:
        value = np.asarray(mask, dtype=np.float64)
        if value.shape != self.dendrite_mask.shape:
            raise ValueError(f"branch mask must have shape {self.dendrite_mask.shape}")
        self.dendrite_mask[...] = np.clip(value, 0.0, 1.0)

    def set_output_authority(self, authority: np.ndarray,
                             simulator_delta: np.ndarray | None = None) -> None:
        value = np.asarray(authority, dtype=np.float64)
        if value.shape != self.network_authority.shape:
            raise ValueError(
                f"output authority must have shape {self.network_authority.shape}")
        self.network_authority[...] = np.clip(value, 0.0, 1.0)
        if simulator_delta is not None:
            fallback = np.asarray(simulator_delta, dtype=np.float64)
            if fallback.shape != self.simulator_delta.shape:
                raise ValueError(
                    f"simulator delta must have shape {self.simulator_delta.shape}")
            self.simulator_delta[...] = fallback

    def forward(self, x: np.ndarray) -> np.ndarray:
        execution = prepare_artifact_execution(
            self.compiled.forward.artifact, self._bindings(x)
        ).run()
        output_id = self.compiled.forward.output_value_ids[0]
        return execution.buffers[output_id].copy()

    def step(self, x: np.ndarray, target: np.ndarray,
             sample_weight: np.ndarray | None = None) -> NativeAdamStep:
        prediction = self.forward(x)
        error = prediction - np.asarray(target, dtype=np.float64)
        if sample_weight is None:
            weight = np.ones((error.shape[0], 1), dtype=np.float64)
        else:
            weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1)
            if weight.shape[0] != error.shape[0] or np.any(weight < 0.0):
                raise ValueError("sample_weight must be nonnegative with one value per row")
        denominator = float(weight.sum() * error.shape[1])
        if denominator <= 0.0:
            raise ValueError("sample_weight must select at least one row")
        loss = float(np.sum(weight * error * error) / denominator)
        seed = 2.0 * weight * error / denominator
        bindings = dict(self._bindings(x))
        bindings[self.compiled.contract.prediction_adjoint.value_id] = seed
        reverse = prepare_artifact_execution(
            self.compiled.forward_vjp.artifact, bindings
        ).run()
        self.iteration += 1
        gradient_sq = 0.0
        for name in self.parameter_names:
            gradient = reverse.buffers[self.gradient_ids[name]]
            gradient_sq += float(np.sum(gradient * gradient))
            first = self.first_moment[name]
            second = self.second_moment[name]
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * gradient * gradient
            first_hat = first / (1.0 - self.beta1 ** self.iteration)
            second_hat = second / (1.0 - self.beta2 ** self.iteration)
            self.parameters[name] -= (
                self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)
            )
        parameter_sq = sum(float(np.sum(value * value))
                           for value in self.parameters.values())
        return NativeAdamStep(
            loss, prediction, gradient_sq ** 0.5, parameter_sq ** 0.5
        )

    def run_cycle(
        self,
        x: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        dendrite_mask: np.ndarray,
        *,
        steps: int | None = None,
    ) -> tuple[NativeAdamStep, np.ndarray]:
        """Run a complete changing-minibatch training bank inside LLVM.

        The host supplies banks and a step count once. Forward, normalized
        weighted loss, generated VJP, accumulation, global norm clipping,
        and Adam state mutation all occur in the native entry.
        """
        cycle = self.training_cycle
        if cycle is None:
            raise RuntimeError("learner was not compiled with a training cycle")
        x = np.ascontiguousarray(x, dtype=np.float64)
        target = np.ascontiguousarray(target, dtype=np.float64)
        weight = np.ascontiguousarray(sample_weight, dtype=np.float64)
        mask = np.ascontiguousarray(dendrite_mask, dtype=np.float64)
        expected_steps = cycle.cycle_length
        if x.shape[0] != expected_steps or target.shape[0] != expected_steps:
            raise ValueError(f"training banks must contain {expected_steps} motions")
        if weight.shape != (expected_steps, x.shape[1], 1):
            raise ValueError("sample_weight bank must have shape (cycle, batch, 1)")
        expected_mask = (expected_steps, *self.dendrite_mask.shape)
        if mask.shape != expected_mask:
            raise ValueError(f"dendrite_mask bank must have shape {expected_mask}")
        if np.any(weight < 0.0) or np.any(weight.sum(axis=1) <= 0.0):
            raise ValueError("every motion must select at least one nonnegative row")
        selected_steps = expected_steps if steps is None else int(steps)
        if not 0 <= selected_steps <= np.iinfo(np.int32).max:
            raise ValueError("steps must fit a nonnegative int32")
        out_dim = target.shape[2]
        loss_scale = np.empty_like(target)
        loss_scale[...] = 1.0 / (weight.sum(axis=1, keepdims=True) * out_dim)
        ids = cycle.input_value_ids
        state = cycle.optimizer_state_value_ids
        values = {
            ids["x"]: x,
            ids["target"]: target,
            ids["sample_weight"]: weight,
            ids["loss_scale"]: loss_scale,
            ids["dendrite_route"]: self.route,
            ids["dendrite_mask"]: mask,
            ids["network_authority"]: self.network_authority,
            ids["simulator_delta"]: self.simulator_delta,
            state["steps"]: np.asarray(selected_steps, dtype=np.int32),
            state["learning_rate"]: np.asarray(self.learning_rate),
            state["beta1"]: np.asarray(self.beta1),
            state["beta2"]: np.asarray(self.beta2),
            state["epsilon"]: np.asarray(self.epsilon),
            state["beta1_power"]: np.asarray(self.beta1_power),
            state["beta2_power"]: np.asarray(self.beta2_power),
            state["iteration"]: np.asarray(self.iteration, dtype=np.int32),
            state["gradient_norm"]: np.asarray(0.0),
            state["clipped_gradient_norm"]: np.asarray(0.0),
        }
        for name in self.parameter_names:
            parameter_id = cycle.parameter_value_ids[name]
            values[parameter_id] = self.parameters[name]
            values[state["first_moment"][parameter_id]] = self.first_moment[name]
            values[state["second_moment"][parameter_id]] = self.second_moment[name]
            values[state["gradient_accumulator"][parameter_id]] = np.zeros_like(
                self.parameters[name])
        execution = prepare_artifact_execution(cycle.artifact, values).run()
        for name in self.parameter_names:
            parameter_id = cycle.parameter_value_ids[name]
            self.parameters[name][...] = execution.buffers[parameter_id]
            self.first_moment[name][...] = execution.buffers[
                state["first_moment"][parameter_id]]
            self.second_moment[name][...] = execution.buffers[
                state["second_moment"][parameter_id]]
        self.iteration = int(execution.buffers[state["iteration"]])
        self.beta1_power = float(execution.buffers[state["beta1_power"]])
        self.beta2_power = float(execution.buffers[state["beta2_power"]])
        loss_history = execution.buffers[
            cycle.output_value_ids["loss_0"]].reshape(-1).copy()
        prediction = execution.buffers[
            cycle.output_value_ids["prediction"]].copy()
        parameter_sq = sum(float(np.sum(value * value))
                           for value in self.parameters.values())
        result = NativeAdamStep(
            float(loss_history[(selected_steps - 1) % expected_steps])
            if selected_steps else 0.0,
            prediction,
            float(execution.buffers[state["clipped_gradient_norm"]]),
            parameter_sq ** 0.5,
        )
        return result, loss_history

    def dendrite_activity(self) -> np.ndarray:
        weights = self.parameters["dendrite_weight"]
        gains = np.abs(self.parameters["dendrite_gain"][0])
        return np.linalg.norm(weights, axis=0) * gains


__all__ = ["CompiledPerforatedAdam", "NativeAdamStep"]
