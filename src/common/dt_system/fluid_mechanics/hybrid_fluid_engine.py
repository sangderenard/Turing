# -*- coding: utf-8 -*-
"""DtCompatibleEngine adapter for Hybrid particle–grid fluid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine
from ..debug import dbg, is_enabled, pretty_metrics


@dataclass
class HybridFluidEngine(DtCompatibleEngine):
    sim: object
    name: str = "bath.hybrid"

    _last_metrics: Optional[Metrics] = None

    def snapshot(self):  # pragma: no cover
        if hasattr(self.sim, "copy_shallow"):
            return self.sim.copy_shallow()
        return None

    def restore(self, snap) -> None:  # pragma: no cover
        if hasattr(self.sim, "restore") and snap is not None:
            self.sim.restore(snap)

    def step(self, dt: float):
        if is_enabled():
            dbg("eng.bath").debug(f"hybrid step: dt={float(dt):.6g} name={self.name}")
        try:
            self.sim.step(float(dt))
        except Exception as e:
            if is_enabled():
                dbg("eng.bath").debug(f"ERROR during hybrid sim.step: {type(e).__name__}: {e}")
            metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=1e9, mass_err=1e9)
            return False, metrics
        try:
            metrics = self._compute_metrics()
        except Exception:
            metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)
        self._last_metrics = metrics
        if is_enabled():
            dbg("eng.bath").debug(f"hybrid done: {pretty_metrics(metrics)}")
        return True, metrics

    def step_with_state(self, state: object, dt: float, *, realtime: bool = False):  # pragma: no cover - light bridge
        try:
            if isinstance(state, dict):
                for k in ("x", "v", "grid"):
                    if k in state and hasattr(self.sim, k):
                        setattr(self.sim, k, state[k])
        except Exception:
            pass
        ok, m = self.step(float(dt))
        new_state = state
        try:
            if isinstance(state, dict):
                out = {}
                for k in ("x", "v", "grid"):
                    if hasattr(self.sim, k):
                        out[k] = getattr(self.sim, k)
                new_state = out
        except Exception:
            new_state = state
        return ok, m, new_state

    def _compute_metrics(self) -> Metrics:
        try:
            grid = self.sim.grid
            vmax_grid = float(max(np.max(np.abs(grid.u)), np.max(np.abs(grid.v)), np.max(np.abs(grid.w))))
        except Exception:
            vmax_grid = 0.0
        try:
            vmax_p = float(np.max(np.linalg.norm(self.sim.v, axis=1))) if getattr(self.sim, "v", None) is not None else 0.0
        except Exception:
            vmax_p = 0.0
        vmax = max(vmax_grid, vmax_p)
        dx = float(getattr(getattr(self.sim, "params", object()), "dx", 1.0))
        try:
            inv_dx = 1.0 / max(dx, 1e-12)
            div = np.zeros_like(self.sim.phi)
            div += (self.sim.grid.u[1:, :, :] - self.sim.grid.u[:-1, :, :]) * inv_dx
            div += (self.sim.grid.v[:, 1:, :] - self.sim.grid.v[:, :-1, :]) * inv_dx
            div += (self.sim.grid.w[:, :, 1:] - self.sim.grid.w[:, :, :-1]) * inv_dx
            div_inf = float(np.max(np.abs(div)))
        except Exception:
            div_inf = 0.0
        mass_err = 0.0
        # Sidechain dt limit hint: min(grid stable dt, particle advective dt)
        dt_hint = None
        try:
            if hasattr(self.sim.grid, "_stable_dt"):
                dt_grid = float(self.sim.grid._stable_dt())  # type: ignore[attr-defined]
            else:
                dt_grid = float("inf")
            vmax_p = float(np.max(np.linalg.norm(self.sim.v, axis=1))) if getattr(self.sim, "v", None) is not None and self.sim.v.size else 0.0
            dx = float(getattr(getattr(self.sim, "params", object()), "dx", 1.0))
            cfl = float(getattr(getattr(self.sim, "params", object()), "cfl", 0.5))
            adv = float("inf") if vmax_p == 0.0 else cfl * dx / vmax_p
            dt_hint = min(dt_grid, adv)
        except Exception:
            dt_hint = None
        return Metrics(max_vel=vmax, max_flux=vmax, div_inf=div_inf, mass_err=mass_err, dt_limit=dt_hint)

    def preferred_dt(self) -> Optional[float]:  # pragma: no cover
        dt0 = getattr(self.sim, "_stable_dt", None)
        try:
            return float(dt0()) if callable(dt0) else None
        except Exception:
            return None

    def get_metrics(self) -> Optional[Metrics]:  # pragma: no cover
        return self._last_metrics


__all__ = ["HybridFluidEngine"]
