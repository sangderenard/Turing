"""Adaptive scientific-reference/network work sharing for tyre hub wrenches."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Sequence

import numpy as np
import sympy

from .ssa_c_backend import CFunctionArtifact, emit_ssa_function_to_c
from .ssa_wasm_backend import SSAWasmArtifact, emit_ssa_function_to_wasm
from .symbolic_equation_compiler import (
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_sympy_equations,
)


@dataclass(frozen=True, slots=True)
class TireForceWorkShareConfig:
    """Loss thresholds and rates for the exact-work duty controller."""

    low_normalized_loss: float = 2.5e-4
    high_normalized_loss: float = 1.0e-2
    loss_ema_rate: float = 0.18
    alpha_rise_rate: float = 0.72
    alpha_fall_rate: float = 0.045
    minimum_reference_alpha: float = 0.02
    maximum_trial_interval: int = 64
    override_trigger: float = 0.025

    def __post_init__(self) -> None:
        values = (
            self.low_normalized_loss, self.high_normalized_loss,
            self.loss_ema_rate, self.alpha_rise_rate, self.alpha_fall_rate,
            self.minimum_reference_alpha, self.override_trigger,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("work-share values must be finite and nonnegative")
        if self.high_normalized_loss <= self.low_normalized_loss:
            raise ValueError("high loss threshold must exceed low loss threshold")
        if any(value > 1 for value in (
            self.loss_ema_rate, self.alpha_rise_rate, self.alpha_fall_rate,
            self.minimum_reference_alpha, self.override_trigger,
        )):
            raise ValueError("work-share rates and alphas cannot exceed one")
        if self.maximum_trial_interval <= 0:
            raise ValueError("maximum trial interval must be positive")


@dataclass(slots=True)
class TireForceWorkShareState:
    alpha: float = 1.0
    normalized_loss_ema: float = 1.0
    trial_phase: float = 0.0
    steps_since_reference: int = 0
    reference_trials: int = 0
    last_trial_loss: float = math.inf
    last_effective_alpha: float = 1.0


def normalized_hub_wrench_loss(
    predicted: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    scale: Sequence[float] | np.ndarray,
) -> float:
    predicted_array = np.asarray(predicted, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if predicted_array.shape != reference_array.shape:
        raise ValueError("predicted and reference hub wrenches must share shape")
    if predicted_array.shape[-1:] != (6,) or scale_array.shape != (6,):
        raise ValueError("hub wrench loss requires six outputs and six scales")
    if np.any(~np.isfinite(scale_array)) or np.any(scale_array <= 0):
        raise ValueError("hub wrench scales must be finite and positive")
    residual = (predicted_array - reference_array) / scale_array
    return float(np.mean(residual * residual))


def _positive(value: sympy.Basic) -> sympy.Basic:
    return (value + sympy.Abs(value)) / 2


def _clamp01(value: sympy.Basic) -> sympy.Basic:
    return sympy.Min(sympy.Max(value, 0), 1)


def symbolic_tire_force_workshare_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Shared loss-to-alpha transition, branch-free and backend portable."""

    names = (
        "previous_alpha previous_loss_ema trial_loss trial_performed "
        "low_loss high_loss loss_ema_rate alpha_rise_rate alpha_fall_rate "
        "minimum_reference_alpha plastic_activity contact_novelty"
        " thermodynamic_novelty"
    )
    s = {name: sympy.Symbol(name, real=True) for name in names.split()}
    performed = _clamp01(s["trial_performed"])
    loss_ema = (
        s["previous_loss_ema"]
        + performed * s["loss_ema_rate"]
        * (s["trial_loss"] - s["previous_loss_ema"])
    )
    loss_coordinate = _clamp01(
        (loss_ema - s["low_loss"])
        / (s["high_loss"] - s["low_loss"] + sympy.Float("1e-18"))
    )
    # Smoothstep avoids an alpha derivative jump at either validation gate.
    loss_demand = loss_coordinate ** 2 * (3 - 2 * loss_coordinate)
    override = _clamp01(sympy.Max(
        s["plastic_activity"], s["contact_novelty"], s["thermodynamic_novelty"],
    ))
    target_alpha = sympy.Max(
        s["minimum_reference_alpha"]
        + (1 - s["minimum_reference_alpha"]) * loss_demand,
        override,
    )
    delta = target_alpha - s["previous_alpha"]
    rising = _positive(delta)
    falling = delta - rising
    alpha = _clamp01(
        s["previous_alpha"]
        + s["alpha_rise_rate"] * rising
        + s["alpha_fall_rate"] * falling
    )
    effective_alpha = sympy.Max(alpha, override)
    values = {
        "next_alpha": alpha,
        "next_loss_ema": loss_ema,
        "effective_alpha": effective_alpha,
        "loss_demand": loss_demand,
        "override_demand": override,
    }
    equations = tuple(
        sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
        for name, expression in values.items()
    )
    return equations, s


@lru_cache(maxsize=1)
def compile_tire_force_workshare_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_tire_force_workshare_equations()
    return compile_sympy_equations(
        equations,
        name="tire_force_reference_workshare",
        publications=tuple(
            SymbolicPublication(str(eq.lhs), f"world.vehicle.tire.workshare.{eq.lhs}")
            for eq in equations
        ),
        dtype="float64",
    )


@lru_cache(maxsize=1)
def compile_tire_force_workshare_c() -> CFunctionArtifact:
    compiled = compile_tire_force_workshare_ssa()
    artifact = emit_ssa_function_to_c(
        compiled.module, compiled.function.name,
        entry_name="tire_force_reference_workshare",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"tire work-share controller does not lower to C: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def compile_tire_force_workshare_wasm() -> SSAWasmArtifact:
    compiled = compile_tire_force_workshare_ssa()
    artifact = emit_ssa_function_to_wasm(
        compiled.module, compiled.function.name, work_contract="deploy",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"tire work-share controller does not lower to Wasm: {reasons}")
    return artifact


class TireForceReferenceWorkShare:
    """Deterministic duty-cycle scheduler with periodic real teacher trials."""

    def __init__(
        self,
        *,
        output_scale: Sequence[float],
        config: TireForceWorkShareConfig = TireForceWorkShareConfig(),
    ) -> None:
        self.output_scale = np.asarray(output_scale, dtype=np.float64)
        if self.output_scale.shape != (6,) or np.any(self.output_scale <= 0):
            raise ValueError("work share requires six positive wrench scales")
        self.config = config
        self.state = TireForceWorkShareState()

    @staticmethod
    def _override(
        plastic_activity: float,
        contact_novelty: float,
        thermodynamic_novelty: float,
    ) -> float:
        return float(np.clip(max(
            plastic_activity, contact_novelty, thermodynamic_novelty,
        ), 0.0, 1.0))

    def reference_due(
        self, *, plastic_activity: float = 0.0, contact_novelty: float = 0.0,
        thermodynamic_novelty: float = 0.0,
    ) -> bool:
        """Advance the exact-work budget and decide whether to run the teacher."""

        override = self._override(plastic_activity, contact_novelty, thermodynamic_novelty)
        self.state.steps_since_reference += 1
        if self.state.reference_trials == 0:
            return True
        if override >= self.config.override_trigger:
            return True
        if self.state.steps_since_reference >= self.config.maximum_trial_interval:
            return True
        self.state.trial_phase += self.state.alpha
        if self.state.trial_phase >= 1.0:
            self.state.trial_phase -= 1.0
            return True
        return False

    def observe_trial(
        self,
        predicted: Sequence[float] | np.ndarray,
        reference: Sequence[float] | np.ndarray,
        *,
        plastic_activity: float = 0.0,
        contact_novelty: float = 0.0,
        thermodynamic_novelty: float = 0.0,
    ) -> float:
        """Update validation loss and exact-work share from one real trial."""

        loss = normalized_hub_wrench_loss(predicted, reference, self.output_scale)
        cfg, state = self.config, self.state
        state.normalized_loss_ema += cfg.loss_ema_rate * (
            loss - state.normalized_loss_ema
        )
        coordinate = float(np.clip(
            (state.normalized_loss_ema - cfg.low_normalized_loss)
            / (cfg.high_normalized_loss - cfg.low_normalized_loss),
            0.0, 1.0,
        ))
        demand = coordinate * coordinate * (3.0 - 2.0 * coordinate)
        override = self._override(plastic_activity, contact_novelty, thermodynamic_novelty)
        target = max(
            cfg.minimum_reference_alpha
            + (1.0 - cfg.minimum_reference_alpha) * demand,
            override,
        )
        delta = target - state.alpha
        rate = cfg.alpha_rise_rate if delta >= 0 else cfg.alpha_fall_rate
        state.alpha = float(np.clip(state.alpha + rate * delta, 0.0, 1.0))
        state.last_effective_alpha = max(state.alpha, override)
        state.last_trial_loss = loss
        state.steps_since_reference = 0
        state.reference_trials += 1
        return loss

    def mix(
        self,
        predicted: Sequence[float] | np.ndarray,
        reference: Sequence[float] | np.ndarray | None,
        *,
        plastic_activity: float = 0.0,
        contact_novelty: float = 0.0,
        thermodynamic_novelty: float = 0.0,
    ) -> np.ndarray:
        prediction = np.asarray(predicted, dtype=np.float64)
        if prediction.ndim < 1 or prediction.shape[-1] != 6:
            raise ValueError("work-share prediction must end in a six-axis wrench")
        if reference is None:
            return prediction
        truth = np.asarray(reference, dtype=np.float64)
        if truth.shape != prediction.shape:
            raise ValueError("work-share reference must match the predicted wrench batch")
        effective = max(
            self.state.alpha,
            self._override(plastic_activity, contact_novelty, thermodynamic_novelty),
        )
        self.state.last_effective_alpha = effective
        return effective * truth + (1.0 - effective) * prediction

    def step(
        self,
        predicted: Sequence[float] | np.ndarray,
        reference_evaluator: Callable[[], Sequence[float] | np.ndarray],
        *,
        plastic_activity: float = 0.0,
        contact_novelty: float = 0.0,
        thermodynamic_novelty: float = 0.0,
    ) -> tuple[np.ndarray, bool, float | None]:
        """Run one live step, invoking the expensive teacher only when due."""

        due = self.reference_due(
            plastic_activity=plastic_activity,
            contact_novelty=contact_novelty,
            thermodynamic_novelty=thermodynamic_novelty,
        )
        if not due:
            return self.mix(predicted, None), False, None
        reference = np.asarray(reference_evaluator(), dtype=np.float64)
        loss = self.observe_trial(
            predicted, reference,
            plastic_activity=plastic_activity,
            contact_novelty=contact_novelty,
            thermodynamic_novelty=thermodynamic_novelty,
        )
        return self.mix(
            predicted, reference,
            plastic_activity=plastic_activity,
            contact_novelty=contact_novelty,
            thermodynamic_novelty=thermodynamic_novelty,
        ), True, loss
