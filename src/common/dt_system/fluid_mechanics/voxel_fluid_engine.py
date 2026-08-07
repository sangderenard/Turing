# -*- coding: utf-8 -*-
"""DtCompatibleEngine adapter for Voxel (MAC) fluid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine
from ..debug import dbg, is_enabled, pretty_metrics


@dataclass
class VoxelFluidEngine(DtCompatibleEngine):
    sim: object
    dedup: bool = False
    def register_vertices(self, state_table, positions, masses):
        uuids = []
        for pos, mass in zip(positions, masses):
            uuid_str = state_table.register_identity(pos, mass, dedup=self.dedup)
            uuids.append(uuid_str)
        return uuids
    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        for k in ("u", "v", "w", "pr", "T", "S"):
            if hasattr(self.sim, k):
                out[k] = getattr(self.sim, k)
        return out
    
    name: str = "bath.voxel"

    _last_metrics: Optional[Metrics] = None

    def snapshot(self):  # pragma: no cover
        if hasattr(self.sim, "copy_shallow"):
            return self.sim.copy_shallow()
        return None

    def restore(self, snap) -> None:  # pragma: no cover
        if hasattr(self.sim, "restore") and snap is not None:
            self.sim.restore(snap)

    def step(self, dt: float, state=None, state_table=None):
        if is_enabled():
            dbg("eng.bath").debug(f"voxel step: dt={float(dt):.6g} name={self.name}")
        try:
            # Optionally update sim state from state dict
            if isinstance(state, dict):
                for k in ("u", "v", "w", "pr", "T", "S"):
                    if k in state and hasattr(self.sim, k):
                        setattr(self.sim, k, state[k])
            self.sim.step(float(dt))
        except Exception as e:
            if is_enabled():
                dbg("eng.bath").debug(f"ERROR during voxel sim.step: {type(e).__name__}: {e}")
            metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=1e9, mass_err=1e9)
            return False, metrics, self.get_state()
        try:
            u, v, w = self.sim.u, self.sim.v, self.sim.w
            vmax = float(max(np.max(np.abs(u)), np.max(np.abs(v)), np.max(np.abs(w))))
        except Exception:
            vmax = 0.0
        dt_hint = None
        try:
            if hasattr(self.sim, "_stable_dt"):
                dt_hint = float(self.sim._stable_dt())  # type: ignore[attr-defined]
        except Exception:
            dt_hint = None
        div_inf = 0.0
        try:
            div = (
                (self.sim.u[1:, :, :] - self.sim.u[:-1, :, :])
                + (self.sim.v[:, 1:, :] - self.sim.v[:, :-1, :])
                + (self.sim.w[:, :, 1:] - self.sim.w[:, :, :-1])
            ) / float(self.sim.dx)
            if hasattr(self.sim, "solid"):
                div = np.where(self.sim.solid, 0.0, div)
            div_inf = float(np.max(np.abs(div))) if div.size else 0.0
        except Exception:
            div_inf = 0.0
        metrics = Metrics(max_vel=vmax, max_flux=vmax, div_inf=div_inf, mass_err=0.0, dt_limit=dt_hint)
        self._last_metrics = metrics
        if is_enabled():
            dbg("eng.bath").debug(f"voxel done: {pretty_metrics(metrics)}")
        return True, metrics, self.get_state()

    def step_with_state(self, state: object, dt: float, *, realtime: bool = False):  # pragma: no cover - light bridge
        return self.step(float(dt), state=state)

    def preferred_dt(self) -> Optional[float]:  # pragma: no cover
        dt0 = getattr(self.sim, "_stable_dt", None)
        try:
            return float(dt0()) if callable(dt0) else None
        except Exception:
            return None

    def get_metrics(self) -> Optional[Metrics]:  # pragma: no cover
        return self._last_metrics


__all__ = ["VoxelFluidEngine"]
