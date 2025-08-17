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

    def step_realtime(self, dt: float) -> tuple[bool, Metrics]:  # pragma: no cover - exercised via demo
        if self.runner is None:
            self.runner = MetaLoopRunner()

        # Ensure the runner has a realtime config. If not, create one using
        # the provided dt as the budget.
        if self.runner._realtime_config is None:
            from .realtime import RealtimeConfig, RealtimeState
            # dt is in seconds, budget_ms is in milliseconds
            rt_cfg = RealtimeConfig(budget_ms = dt * 1000.0)
            rt_state = RealtimeState()
            self.runner.set_realtime_config(rt_cfg, rt_state)
        else:
            # Update budget for this frame
            self.runner._realtime_config.budget_ms = dt * 1000.0

        try:
            if is_enabled():
                dbg("roundnode").debug(f"step_realtime: dt={float(dt):.6g} inner={self.inner.label}")
            # We don't pass dt to run_round, because the budget is now in the config.
            res = self.runner.run_round(self.inner, realtime=True)
            m = res.metrics or Metrics(0.0, 0.0, 0.0, 0.0)
            if is_enabled():
                dbg("roundnode").debug(f"step_realtime done: metrics=({pretty_metrics(m)})")
            return True, m
        except Exception as e:
            if is_enabled():
                dbg("roundnode").error(f"step_realtime failed: {e}")
            return False, Metrics(0.0, 0.0, 1e9, 1e9)


__all__ = ["RoundNodeEngine"]
