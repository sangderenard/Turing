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

Until wired, this class raises NotImplementedError in step().
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
    solver: object
    name: str = "softbody.xpbd"
    # Optional providers that deliver surface data (pressure/velocity/etc.)
    surface_samplers: tuple[SurfaceSampler, ...] = ()
    # Optional hook to return a scalar penetration value for dt solving
    penetration_fn: Optional[Callable[[object], float]] = None
    # Optional solids registry to consult for collisions/contacts
    solids: Optional["SolidRegistry"] = None
    _last_metrics: Optional[Metrics] = None

    def snapshot(self):  # pragma: no cover
        if hasattr(self.solver, "copy_shallow"):
            return self.solver.copy_shallow()
        return None

    def restore(self, snap) -> None:  # pragma: no cover
        if hasattr(self.solver, "restore") and snap is not None:
            self.solver.restore(snap)

    def step(self, dt: float, state=None, state_table=None):
      # Optionally update solver state from state dict
      if isinstance(state, dict):
        for k, v in state.items():
          if hasattr(self.solver, k):
            setattr(self.solver, k, v)
      # Optionally use state_table for advanced metrics (not implemented)
      # TODO: implement full step logic
      # For now, just return ok, metrics, and the current state
      metrics = None  # Optionally compute or set a Metrics object if available
      new_state = self.solver.copy_shallow() if hasattr(self.solver, 'copy_shallow') else None
      return True, metrics, new_state

    def preferred_dt(self) -> Optional[float]:  # pragma: no cover
        return None

    def get_metrics(self) -> Optional[Metrics]:  # pragma: no cover
        return self._last_metrics


__all__ = ["SoftbodyEngineWrapper"]
