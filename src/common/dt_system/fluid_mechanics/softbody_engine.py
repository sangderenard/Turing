# -*- coding: utf-8 -*-
"""Stub DtCompatibleEngine wrapper for the softbody simulator (XPBD).

This wrapper will adapt a softbody solver (e.g., XPBDSolver + hierarchy) to the
DtCompatibleEngine interface, enabling orchestration under the dt-graph runner.

Important design notes for a correct implementation
---------------------------------------------------
- Surface coupling: Softbody membranes often need pressure and shear samples
  from other engines (fluids, wind/drag fields). Design the wrapper to accept
  one or more provider callables, e.g. ``sample_surface(points)->{P, v, ...}``,
  and call them during ``step`` before running constraint projection.
- Bidirectional coupling: If the softbody exerts reaction forces back onto the
  fluid/fields, arrange a handshake API (e.g., accumulate impulses/flux and
  forward via an engine-specific sink function) or set up a meta-engine round
  where both wrappers exchange data iteratively per substep.
- Time integration: XPBD typically performs multiple solver iterations inside
  each substep. Keep dt exact at the wrapper level, and iterate internally.
- Metrics: Report stability proxies (max velocity, constraint violation norms,
  penetration depth) via the standard Metrics object. If you can compute a
  robust "penetration" scalar, set it in div_inf so the optional dt solver can
  monotone-bisect it to zero.
- Snapshot/restore: Provide shallow copy helpers so the bisection solver can
  try multiple dt candidates without committing state.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, TYPE_CHECKING
import numpy as np

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine
if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..solids.api import SolidRegistry

SurfaceSampler = Callable[[np.ndarray], Dict[str, np.ndarray]]


@dataclass
class SoftbodyEngineWrapper(DtCompatibleEngine):
    """Generic DtCompatibleEngine adapter for XPBD-style softbody solvers."""

    solver: object
    name: str = "softbody.xpbd"
    surface_samplers: tuple[SurfaceSampler, ...] = ()
    penetration_fn: Optional[Callable[[object], float]] = None
    solids: Optional["SolidRegistry"] = None
    dedup: bool = False
    _last_metrics: Optional[Metrics] = None

    def register_vertices(self, state_table, positions, masses):
        uuids = []
        for pos, mass in zip(positions, masses):
            uuid_str = state_table.register_identity(pos, mass, dedup=self.dedup)
            uuids.append(uuid_str)
        return uuids

    def snapshot(self):  # pragma: no cover
        if hasattr(self.solver, "copy_shallow"):
            try:
                return self.solver.copy_shallow()
            except Exception:
                return None
        return None

    def restore(self, snap) -> None:  # pragma: no cover
        if hasattr(self.solver, "restore") and snap is not None:
            try:
                self.solver.restore(snap)
            except Exception:
                pass

    def _snapshot_state(self):
        if hasattr(self.solver, "copy_shallow"):
            try:
                return self.solver.copy_shallow()
            except Exception:
                return None
        return None

    def _compute_metrics(self) -> Metrics:
        try:
            if hasattr(self.solver, "max_vertex_speed"):
                vmax = float(self.solver.max_vertex_speed())
            elif hasattr(self.solver, "V"):
                vmax = float(np.max(np.linalg.norm(self.solver.V, axis=1)))
            else:
                vmax = 0.0
        except Exception:
            vmax = 0.0
        pen = 0.0
        if self.penetration_fn is not None:
            try:
                pen = float(self.penetration_fn(self.solver))
            except Exception:
                pen = 0.0
        dt_hint = None
        try:
            dt_fn = getattr(self.solver, "_stable_dt", None)
            if callable(dt_fn):
                dt_hint = float(dt_fn())
        except Exception:
            dt_hint = None
        return Metrics(max_vel=vmax, max_flux=vmax, div_inf=pen, mass_err=0.0, dt_limit=dt_hint)

    def step(self, dt: float, state=None, state_table=None):
        if isinstance(state, dict):
            for k, v in state.items():
                if hasattr(self.solver, k):
                    setattr(self.solver, k, v)
        try:
            self.solver.step(float(dt))
        except Exception:
            metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=1e9, mass_err=1e9)
            return False, metrics, self._snapshot_state()
        metrics = self._compute_metrics()
        self._last_metrics = metrics
        new_state = self._snapshot_state()
        return True, metrics, new_state

    def preferred_dt(self) -> Optional[float]:  # pragma: no cover
        dt0 = getattr(self.solver, "_stable_dt", None)
        try:
            return float(dt0()) if callable(dt0) else None
        except Exception:
            return None

    def get_metrics(self) -> Optional[Metrics]:  # pragma: no cover
        return self._last_metrics


__all__ = ["SoftbodyEngineWrapper"]
