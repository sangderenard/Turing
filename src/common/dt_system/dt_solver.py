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
- The engine implements optional ``snapshot()``/``restore(snap)``. If absent
  and ``require_snapshot`` is True, a ValueError is raised.

Usage
-----
- Construct a :class:`BisectSolverConfig` and pass it via EngineRegistration
  (``solver_config=...``) to opt-in for a specific engine, or call
  :func:`solve_window_bisect` directly.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Any

from .dt_scaler import Metrics
from .debug import dbg, is_enabled, pretty_metrics
from .state_table import GLOBAL_STATE_TABLE, sync_engine_from_table, publish_engine_to_table


ObjectiveFn = Callable[[Metrics], float]


@dataclass
class BisectSolverConfig:
    target: float
    eps: float = 1e-6
    dt_min: float = 1e-9
    dt_max: Optional[float] = None  # per micro-step cap; default to remainder
    max_iters: int = 30
    # Choose objective either by field name or callable. If both provided, callable wins.
    field: Optional[str] = None  # e.g., "div_inf", "mass_err", "max_vel", "max_flux"
    objective: Optional[ObjectiveFn] = None
    # Monotonic direction of objective as dt increases: "increase" or "decrease"
    monotonic: str = "increase"
    # Require snapshot/restore; when False, evaluations mutate state and cannot rollback.
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


def _eval_on_snapshot(engine: Any, dt: float, cfg: BisectSolverConfig) -> tuple[bool, Metrics, Any]:
    snap = None
    # Prefer engine-level snapshot/restore; otherwise try common state attrs
    state_holder = None
    if _has_snapshot_api(engine):
        snap = engine.snapshot()
    else:
        for attr in ("s", "sim", "state"):
            if hasattr(engine, attr):
                candidate = getattr(engine, attr)
                if _has_snapshot_api(candidate):
                    state_holder = candidate
                    try:
                        snap = candidate.snapshot()
                    except Exception:
                        snap = None
                    break
        if snap is None and cfg.require_snapshot:
            raise ValueError("No snapshot/restore available for bisect solver (engine or inner state)")

    # Sync from shared table before evaluation
    try:
        sync_engine_from_table(engine, getattr(engine, "name", getattr(engine, "__class__", type(engine)).__name__), GLOBAL_STATE_TABLE)
    except Exception:
        pass
    ok, m = engine.step(dt)
    # Roll back if we can; ignore errors
    try:
        if snap is not None:
            if state_holder is not None:
                state_holder.restore(snap)
            else:
                engine.restore(snap)
    except Exception:
        pass
    return bool(ok), m, snap


def solve_window_bisect(engine: Any, total_dt: float, cfg: BisectSolverConfig) -> Metrics:
    """Advance ``engine`` by ``total_dt`` using bisection micro-steps.

    Returns the Metrics of the final micro-step.
    """
    advanced = 0.0
    last_metrics = Metrics(0.0, 0.0, 0.0, 0.0)
    while (total_dt - advanced) > 1e-15:
        remainder = total_dt - advanced
        dt_lo = max(min(cfg.dt_min, remainder), 1e-30)
        dt_hi = min(cfg.dt_max if cfg.dt_max is not None else remainder, remainder)

        # Evaluate endpoints
        ok_lo, m_lo, _ = _eval_on_snapshot(engine, dt_lo, cfg)
        ok_hi, m_hi, _ = _eval_on_snapshot(engine, dt_hi, cfg)
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
                    ok_mid, m_mid, _ = _eval_on_snapshot(engine, mid, cfg)
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

        # Commit chosen dt
        # Sync from state table before commit, publish after commit
        try:
            sync_engine_from_table(engine, getattr(engine, "name", getattr(engine, "__class__", type(engine)).__name__), GLOBAL_STATE_TABLE)
        except Exception:
            pass
        ok, m_commit = engine.step(pick_dt)
        try:
            publish_engine_to_table(engine, getattr(engine, "name", getattr(engine, "__class__", type(engine)).__name__), GLOBAL_STATE_TABLE)
        except Exception:
            pass
        last_metrics = m_commit if ok else pick_m
        advanced += float(pick_dt)
        if is_enabled():
            dbg("solver").debug(
                f"commit: dt={pick_dt:.6g} advanced={advanced:.6g}/{total_dt:.6g} metrics=({pretty_metrics(last_metrics)})"
            )

    return last_metrics


__all__ = [
    "BisectSolverConfig",
    "solve_window_bisect",
]
