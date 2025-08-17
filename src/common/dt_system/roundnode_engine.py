# -*- coding: utf-8 -*-
"""Adapter: treat a RoundNode as a DtCompatibleEngine.

This is useful when embedding a nested subgraph as a "system node" that
still respects the centralized dt cascade: step(dt) runs one superstep of
the inner RoundNode using dt as the round_max window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .dt_scaler import Metrics
from .engine_api import DtCompatibleEngine
from .dt_graph import MetaLoopRunner, RoundNode
from .debug import dbg, is_enabled, pretty_metrics


@dataclass
class RoundNodeEngine(DtCompatibleEngine):
    inner: RoundNode
    runner: Optional[MetaLoopRunner] = None

    def __post_init__(self) -> None:
        if self.runner is None:
            self.runner = MetaLoopRunner()

    def step(self, dt: float):
        # Temporarily override the inner plan to match the requested slice
        saved = self.inner.plan
        from .dt import SuperstepPlan
        self.inner.plan = SuperstepPlan(round_max=float(dt), dt_init=max(float(dt), 1e-12))
        try:
            if is_enabled():
                dbg("roundnode").debug(f"step: dt={float(dt):.6g} inner={self.inner.label}")
            res = self.runner.run_round(self.inner)
        finally:
            self.inner.plan = saved
        m = res.metrics or Metrics(0.0, 0.0, 0.0, 0.0)
        if is_enabled():
            dbg("roundnode").debug(f"step done: metrics=({pretty_metrics(m)})")
        return True, m

    # Lightweight realtime path: advance children once without inner supersteps.
    # Used by realtime mode to avoid nested time control.
    def step_realtime(self, dt: float) -> tuple[bool, Metrics]:  # pragma: no cover - exercised via demo
        try:
            from .dt_scaler import Metrics as _M
            from .dt_graph import SuperstepPlan as _Plan  # type: ignore
        except Exception:
            pass

        # Construct a one-off adapter similar to MetaLoopRunner._advance_children
        # but without recursive superstep execution.
        last_m: Optional[Metrics] = None
        if is_enabled():
            dbg("roundnode").debug(f"step_realtime: dt={float(dt):.6g} inner={self.inner.label}")
        for ch in list(self.inner.children):
            if hasattr(ch, "advance"):
                # Advance leaf once
                ok, m, _state_new = ch.advance(ch.state.state, float(dt))  # type: ignore[attr-defined]
                if not ok:
                    return False, Metrics(0.0, 0.0, 1e9, 1e9)
                last_m = m
            else:
                # Nested round: run its advance children directly
                try:
                    for gch in list(getattr(ch, "children", [])):
                        if hasattr(gch, "advance"):
                            ok, m, _state_new = gch.advance(gch.state.state, float(dt))  # type: ignore[attr-defined]
                            if not ok:
                                return False, Metrics(0.0, 0.0, 1e9, 1e9)
                            last_m = m
                except Exception:
                    pass
        if last_m is None:
            last_m = Metrics(0.0, 0.0, 0.0, 0.0)
        if is_enabled():
            dbg("roundnode").debug(f"step_realtime done: metrics=({pretty_metrics(last_m)})")
        return True, last_m


__all__ = ["RoundNodeEngine"]
