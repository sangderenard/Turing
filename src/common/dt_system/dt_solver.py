from __future__ import annotations

"""Opt-in dt solver: binary search to match a target metric value per micro-step.

This module provides a monotonic bisection-based timestep solver that, for a
given time slice, repeatedly finds a micro-step ``dt`` such that a scalar
objective derived from Metrics matches a prescribed target within ``eps``.
Each candidate evaluation runs on a snapshot of the engine's state and is
rolled back before the next try; the accepted ``dt`` is then committed by
re-running the step on the live engine state. The process repeats until the
parent-required duration is met.

Assumptions
-----------
- The objective is monotonic in ``dt`` over the bracket [dt_lo, dt_hi].
- The engine implements ``snapshot()``/``restore(snap)`` either directly or on
  a conventional inner ``s``, ``sim``, or ``state`` object. Non-transactional
  candidate evaluation is rejected.

Usage
-----
- Construct a :class:`BisectSolverConfig` and pass it via EngineRegistration
  (``solver_config=...``) to opt-in for a specific engine, or call
  :func:`solve_window_bisect` directly.
"""

from dataclasses import dataclass
import inspect
from typing import Callable, Optional, Any

from .dt_scaler import Metrics, coerce_metrics
from .debug import dbg, is_enabled, pretty_metrics
from .state_table import sync_engine_from_table, publish_engine_to_table
from .state_table import StateTable


ObjectiveFn = Callable[[Metrics], float]


@dataclass
class BisectSolverConfig:
    target: float
    eps: float = 1e-6
    dt_min: float = 1e-9
    dt_max: Optional[float] = None  # per micro-step cap; default to remainder
    max_iters: int = 30
    max_steps: int = 100_000
    # Choose objective either by field name or callable. If both provided, callable wins.
    field: Optional[str] = None  # e.g., "div_inf", "mass_err", "max_vel", "max_flux"
    objective: Optional[ObjectiveFn] = None
    # Monotonic direction of objective as dt increases: "increase" or "decrease"
    monotonic: str = "increase"
    # Retained for constructor compatibility. Scientific bisection rejects
    # False because candidate evaluations must never leak into committed state.
    require_snapshot: bool = True


def _get_objective_value(m: Metrics, cfg: BisectSolverConfig) -> float:
    if cfg.objective is not None:
        return float(cfg.objective(m))
    if cfg.field:
        try:
            return float(getattr(m, cfg.field))
        except Exception:
            pass
    # Default objective: use div_inf
    return float(getattr(m, "div_inf", 0.0))


def _has_snapshot_api(obj: Any) -> bool:
    return hasattr(obj, "snapshot") and hasattr(obj, "restore")


def _engine_checkpoint(engine: Any, *, required: bool) -> tuple[Any, Any, dict[str, Any]]:
    """Capture the engine or its conventional inner state plus clock fields."""

    holder = None
    snapshot = None
    if _has_snapshot_api(engine):
        holder = engine
        snapshot = engine.snapshot()
    else:
        for attr in ("s", "sim", "state"):
            candidate = getattr(engine, attr, None)
            if _has_snapshot_api(candidate):
                holder = candidate
                snapshot = candidate.snapshot()
                break
    if snapshot is None and required:
        raise ValueError(
            "No snapshot/restore available for bisect solver "
            "(engine or conventional inner state)"
        )
    clocks = {
        name: getattr(engine, name)
        for name in ("world_time", "observer_time")
        if hasattr(engine, name)
    }
    return holder, snapshot, clocks


def _restore_engine(
    engine: Any,
    checkpoint: tuple[Any, Any, dict[str, Any]],
) -> None:
    holder, snapshot, clocks = checkpoint
    if holder is not None and snapshot is not None:
        holder.restore(snapshot)
    for name, value in clocks.items():
        setattr(engine, name, value)


def _call_engine_step(
    engine: Any,
    dt: float,
    state_table: StateTable,
) -> tuple[bool, Metrics]:
    """Call the engine's declared step signature without masking body errors."""

    step = getattr(engine, "step", None)
    if not callable(step):
        raise TypeError("bisect solver engine must provide step()")
    parameters = inspect.signature(step).parameters
    kwargs: dict[str, Any] = {}
    if "state" in parameters:
        kwargs["state"] = None
    if "state_table" in parameters:
        kwargs["state_table"] = state_table
    result = step(float(dt), **kwargs)
    if not isinstance(result, tuple) or len(result) < 2:
        raise TypeError("engine step must return at least (ok, Metrics)")
    ok, metrics = result[0], result[1]
    return bool(ok), coerce_metrics(metrics)


def _advance_once(
    engine: Any,
    dt: float,
    state_table: StateTable,
    registration_name: str,
) -> tuple[bool, Metrics]:
    sync_engine_from_table(engine, registration_name, state_table)
    ok, metrics = _call_engine_step(engine, dt, state_table)
    if ok:
        publish_engine_to_table(engine, registration_name, state_table)
    return ok, metrics


def _eval_on_snapshot(
    engine: Any,
    dt: float,
    cfg: BisectSolverConfig,
    *,
    state_table: StateTable,
    registration_name: str,
) -> tuple[bool, Metrics]:
    engine_checkpoint = _engine_checkpoint(
        engine, required=cfg.require_snapshot
    )
    table_checkpoint = state_table.snapshot()
    try:
        return _advance_once(
            engine, dt, state_table, registration_name
        )
    finally:
        _restore_engine(engine, engine_checkpoint)
        state_table.restore(table_checkpoint)


def solve_window_bisect(
    engine: Any,
    total_dt: float,
    cfg: BisectSolverConfig,
    *,
    state_table: StateTable,
    registration_name: str = "engine",
) -> Metrics:
    """Advance ``engine`` by ``total_dt`` using bisection micro-steps.

    Returns the Metrics of the final micro-step.
    """
    if state_table is None:
        raise ValueError("bisect solver requires an explicit StateTable")
    if not str(registration_name).strip():
        raise ValueError("bisect solver requires a registration name")
    if not total_dt >= 0.0:
        raise ValueError("bisect solver total_dt must be non-negative")
    if cfg.dt_min <= 0.0:
        raise ValueError("bisect solver dt_min must be positive")
    if cfg.dt_max is not None and cfg.dt_max <= 0.0:
        raise ValueError("bisect solver dt_max must be positive")
    if int(cfg.max_iters) < 1 or int(cfg.max_steps) < 1:
        raise ValueError("bisect solver iteration budgets must be positive")
    if cfg.monotonic not in ("increase", "decrease"):
        raise ValueError("bisect solver monotonic must be increase or decrease")
    if not cfg.require_snapshot:
        raise ValueError(
            "bisect solver requires transactional candidate evaluation"
        )

    advanced = 0.0
    steps = 0
    last_metrics = Metrics(0.0, 0.0, 0.0, 0.0)
    while (total_dt - advanced) > 1e-15:
        steps += 1
        if steps > int(cfg.max_steps):
            raise RuntimeError(
                "bisect solver exceeded its committed microstep budget"
            )
        remainder = total_dt - advanced
        dt_lo = max(min(cfg.dt_min, remainder), 1e-30)
        dt_hi = min(cfg.dt_max if cfg.dt_max is not None else remainder, remainder)

        # Evaluate endpoints
        ok_lo, m_lo = _eval_on_snapshot(
            engine,
            dt_lo,
            cfg,
            state_table=state_table,
            registration_name=registration_name,
        )
        ok_hi, m_hi = _eval_on_snapshot(
            engine,
            dt_hi,
            cfg,
            state_table=state_table,
            registration_name=registration_name,
        )
        f_lo = _get_objective_value(m_lo, cfg) if ok_lo else float("inf")
        f_hi = _get_objective_value(m_hi, cfg) if ok_hi else float("inf")

        if is_enabled():
            dbg("solver").debug(
                f"bisect: rem={remainder:.6g} lo={dt_lo:.6g} f_lo={f_lo:.3e} hi={dt_hi:.6g} f_hi={f_hi:.3e} target={cfg.target:.3e}"
            )

        # Helper to check closeness
        def close(val: float) -> bool:
            return abs(val - cfg.target) <= cfg.eps

        direction = (cfg.monotonic or "increase").lower()

        # If endpoints already satisfy or bracket poorly, choose closest endpoint
        pick_dt = None
        pick_m = None
        if close(f_lo) or close(f_hi):
            pick_dt, pick_m = (dt_lo, m_lo) if close(f_lo) else (dt_hi, m_hi)
        else:
            # Determine which side to move based on direction and target
            def left_is_below():
                return f_lo <= cfg.target if direction == "increase" else f_lo >= cfg.target

            def right_is_above():
                return f_hi >= cfg.target if direction == "increase" else f_hi <= cfg.target

            bracketed = left_is_below() and right_is_above()
            if not bracketed:
                # Can't bracket: choose the closer endpoint
                if abs(f_lo - cfg.target) <= abs(f_hi - cfg.target):
                    pick_dt, pick_m = dt_lo, m_lo
                else:
                    pick_dt, pick_m = dt_hi, m_hi
            else:
                # Standard bisection
                lo, hi = dt_lo, dt_hi
                f_l, f_h = f_lo, f_hi
                m_mid = m_hi
                for _ in range(int(cfg.max_iters)):
                    mid = 0.5 * (lo + hi)
                    ok_mid, m_mid = _eval_on_snapshot(
                        engine,
                        mid,
                        cfg,
                        state_table=state_table,
                        registration_name=registration_name,
                    )
                    f_mid = _get_objective_value(m_mid, cfg) if ok_mid else float("inf")
                    if is_enabled():
                        dbg("solver").debug(
                            f"  mid={mid:.6g} f_mid={f_mid:.3e}"
                        )
                    if close(f_mid):
                        pick_dt, pick_m = mid, m_mid
                        break
                    # Decide which half to keep
                    below = f_mid <= cfg.target if direction == "increase" else f_mid >= cfg.target
                    if below:
                        lo, f_l = mid, f_mid
                    else:
                        hi, f_h = mid, f_mid
                else:
                    # Max iters reached: pick midpoint
                    pick_dt, pick_m = 0.5 * (lo + hi), m_mid

        if pick_dt is None:
            raise RuntimeError("bisect solver failed to select a timestep")
        ok_commit, committed_metrics = _advance_once(
            engine,
            float(pick_dt),
            state_table,
            registration_name,
        )
        if not ok_commit:
            raise RuntimeError(
                f"bisect solver selected dt={pick_dt:.6g} but commit failed"
            )
        advanced += float(pick_dt)
        last_metrics = committed_metrics

    return last_metrics


__all__ = [
    "BisectSolverConfig",
    "solve_window_bisect",
]
