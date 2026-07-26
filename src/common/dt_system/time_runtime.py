"""Stable managed-time boundary for external process coordinators.

The runtime accepts an absolute requested time window, converts authored event
times into exact relative superstep boundaries, and delegates microstep choice
to Turing's adaptive controller.  It is intentionally unaware of camera,
optical, fluid, or Nodus semantics.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable

from .dt import SuperstepPlan, SuperstepResult
from .dt_controller import STController, Targets, run_superstep_plan
from .dt_scaler import Metrics


ManagedAdvance = Callable[[Any, float], tuple[bool, Metrics]]
ManagedCommitGate = Callable[["TimeWindowRequest", SuperstepResult], bool]


def _time_close(left: float, right: float, *, eps: float = 1.0e-15) -> bool:
    scale_tolerance = max(
        math.ulp(float(left)),
        math.ulp(float(right)),
    ) * 8.0
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=max(float(eps), scale_tolerance),
    )


@dataclass(frozen=True)
class TimeWindowRequest:
    request_id: int
    generation: int
    t_start: float
    t_end: float
    dt_initial: float
    event_times: tuple[float, ...] = ()
    allow_increase_mid_window: bool = False

    def validate(self) -> None:
        values = (self.t_start, self.t_end, self.dt_initial, *self.event_times)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("managed-time request values must be finite")
        if int(self.request_id) < 0 or int(self.generation) < 0:
            raise ValueError("managed-time request identity must be non-negative")
        if not self.t_end > self.t_start:
            raise ValueError("managed-time request must have a positive window")
        if self.dt_initial <= 0.0:
            raise ValueError("managed-time initial dt must be positive")
        if tuple(sorted(set(self.event_times))) != self.event_times:
            raise ValueError("managed-time event times must be unique and ordered")
        if any(
            not self.t_start < event < self.t_end
            for event in self.event_times
        ):
            raise ValueError("managed-time events must lie inside the window")

    @property
    def duration(self) -> float:
        return float(self.t_end - self.t_start)

    @property
    def relative_event_boundaries(self) -> tuple[float, ...]:
        return tuple(float(event - self.t_start) for event in self.event_times)


@dataclass(frozen=True)
class TimeAdvanceReport:
    request_id: int
    generation: int
    t_start: float
    t_end: float
    result: SuperstepResult

    @property
    def exact_landing(self) -> bool:
        return _time_close(
            float(self.result.advanced),
            float(self.t_end - self.t_start),
        )


class ManagedTimeRuntime:
    """Scientific managed-time process with revision and transaction checks."""

    def __init__(
        self,
        state: Any,
        advance: ManagedAdvance,
        *,
        dx: float,
        targets: Targets,
        controller: STController | None = None,
        generation: int = 0,
        initial_time: float = 0.0,
    ) -> None:
        if not (
            hasattr(state, "copy_shallow")
            and callable(state.copy_shallow)
            and hasattr(state, "restore")
            and callable(state.restore)
        ):
            raise TypeError(
                "managed scientific state requires copy_shallow()/restore()"
            )
        if not callable(advance):
            raise TypeError("managed-time advance callback is required")
        if not math.isfinite(float(dx)) or float(dx) <= 0.0:
            raise ValueError("managed-time dx must be finite and positive")
        if int(generation) < 0 or not math.isfinite(float(initial_time)):
            raise ValueError("managed-time generation/time are invalid")
        self.state = state
        self.advance_callback = advance
        self.dx = float(dx)
        self.targets = targets
        self.controller = controller or STController()
        self.generation = int(generation)
        self.current_time = float(initial_time)
        self.last_report: TimeAdvanceReport | None = None
        self.last_request_id: int | None = None

    def advance(
        self,
        request: TimeWindowRequest,
        *,
        commit_gate: ManagedCommitGate | None = None,
    ) -> TimeAdvanceReport:
        request.validate()
        if request.generation != self.generation:
            raise RuntimeError("stale managed-time generation")
        if (
            self.last_request_id is not None
            and request.request_id <= self.last_request_id
        ):
            raise RuntimeError("managed-time request was replayed or reordered")
        if not _time_close(
            request.t_start,
            self.current_time,
        ):
            raise RuntimeError(
                "managed-time request does not start at current committed time"
            )

        state_checkpoint = self.state.copy_shallow()
        controller_checkpoint = copy.deepcopy(vars(self.controller))
        plan = SuperstepPlan(
            round_max=request.duration,
            dt_init=request.dt_initial,
            allow_increase_mid_round=request.allow_increase_mid_window,
            event_boundaries=request.relative_event_boundaries,
        )
        try:
            result = run_superstep_plan(
                self.state,
                plan,
                self.dx,
                self.targets,
                self.controller,
                self.advance_callback,
            )
            if not _time_close(
                float(result.advanced),
                request.duration,
                eps=plan.eps,
            ):
                raise RuntimeError("managed-time advance did not land exactly")
            if commit_gate is not None and not commit_gate(request, result):
                raise RuntimeError("managed-time commit gate rejected window")
        except Exception:
            restore_failure: Exception | None = None
            try:
                self.state.restore(state_checkpoint)
            except Exception as exc:
                restore_failure = exc
            finally:
                vars(self.controller).clear()
                vars(self.controller).update(controller_checkpoint)
            if restore_failure is not None:
                raise RuntimeError(
                    "managed-time state rollback failed"
                ) from restore_failure
            raise

        self.current_time = float(request.t_end)
        report = TimeAdvanceReport(
            request_id=request.request_id,
            generation=request.generation,
            t_start=request.t_start,
            t_end=request.t_end,
            result=result,
        )
        self.last_report = report
        self.last_request_id = request.request_id
        return report

    def supersede(self, generation: int, *, at_time: float | None = None) -> None:
        if int(generation) <= self.generation:
            raise ValueError("managed-time generation must increase")
        self.generation = int(generation)
        if at_time is not None:
            if not math.isfinite(float(at_time)):
                raise ValueError("managed-time supersession time must be finite")
            self.current_time = float(at_time)
        self.last_report = None
        self.last_request_id = None


__all__ = [
    "ManagedAdvance",
    "ManagedCommitGate",
    "TimeWindowRequest",
    "TimeAdvanceReport",
    "ManagedTimeRuntime",
]
