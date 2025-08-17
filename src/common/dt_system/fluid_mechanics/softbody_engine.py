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
from typing import Optional, Callable, Dict, Any
import numpy as np

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine

SurfaceSampler = Callable[[np.ndarray], Dict[str, np.ndarray]]


@dataclass
class SoftbodyEngineWrapper(DtCompatibleEngine):
    solver: object
    name: str = "softbody.xpbd"
    # Optional providers that deliver surface data (pressure/velocity/etc.)
    surface_samplers: tuple[SurfaceSampler, ...] = ()
    # Optional hook to return a scalar penetration value for dt solving
    penetration_fn: Optional[Callable[[object], float]] = None

    _last_metrics: Optional[Metrics] = None

    def snapshot(self):  # pragma: no cover
        if hasattr(self.solver, "copy_shallow"):
            return self.solver.copy_shallow()
        return None

    def restore(self, snap) -> None:  # pragma: no cover
        if hasattr(self.solver, "restore") and snap is not None:
            self.solver.restore(snap)

    def step(self, dt: float):
        # Pseudocode for a future full implementation:
        # 1) Sample surface queries from providers (if any):
        #    points = self._surface_points_world()
        #    samples = [sampler(points) for sampler in self.surface_samplers]
        # 2) Feed samples into the solver (pressure forces, drag, etc.)
        # 3) Run solver substeps/iterations for exactly dt.
        # 4) Compute metrics: max velocity, penetration, mass/energy proxies.
        # For now, we provide a safe stub that signals "not yet implemented".
        raise NotImplementedError(
            "SoftbodyEngineWrapper.step is a stub. Wire surface sampling, XPBD iterations, and metrics.\n"
            "Guidance: provide SurfaceSampler providers and a penetration_fn so dt_solver can bisect on it."
        )

    def preferred_dt(self) -> Optional[float]:  # pragma: no cover
        return None

    def get_metrics(self) -> Optional[Metrics]:  # pragma: no cover
        return self._last_metrics


__all__ = ["SoftbodyEngineWrapper"]
