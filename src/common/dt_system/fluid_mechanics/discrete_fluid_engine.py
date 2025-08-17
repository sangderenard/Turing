# -*- coding: utf-8 -*-
"""DtCompatibleEngine adapter for Bath's DiscreteFluid (SPH).

Relocated from ``dt_system/bath_fluid_engine.py`` for consistency with
other fluid mechanics adapters. The class name and API are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine
from ..debug import dbg, is_enabled, pretty_metrics


@dataclass
class BathDiscreteFluidEngine(DtCompatibleEngine):
    """Adapter over :class:`DiscreteFluid` exposing DtCompatibleEngine.

    Parameters
    ----------
    sim:
        A DiscreteFluid instance (from ``src.cells.bath.discrete_fluid``).
    name:
        Optional display name (useful for logs/graphs).
    """

    sim: object
    name: str = "bath.discrete"

    _last_metrics: Optional[Metrics] = None

    # Optional snapshot/restore for bisect solver: delegate to underlying sim if present
    def snapshot(self):  # pragma: no cover
        sim = self.sim
        if hasattr(sim, "snapshot"):
            return sim.snapshot()
        # Conservative snapshot for common arrays if attributes exist
        try:
            import numpy as _np

            return {
                k: _np.copy(getattr(sim, k))
                for k in ("x", "v", "m", "rho", "T", "S")
                if hasattr(sim, k)
            }
        except Exception:
            return None

    def restore(self, snap) -> None:  # pragma: no cover
        sim = self.sim
        try:
            if hasattr(sim, "restore"):
                sim.restore(snap)
                return
            if isinstance(snap, dict):
                for k, arr in snap.items():
                    if hasattr(sim, k):
                        setattr(sim, k, arr)
        except Exception:
            pass

    def step(self, dt: float):
        if is_enabled():
            try:
                N = int(getattr(self.sim, "N", 0))
            except Exception:
                N = 0
            dbg("eng.bath").debug(f"step: dt={float(dt):.6g} name={self.name} N={N}")
        # Save mass to compute conservation error
        try:
            m0 = float(np.sum(self.sim.m))
        except Exception:
            m0 = 0.0

        # Advance exactly dt (sim may internally substep to remain stable)
        try:
            self.sim.step(float(dt))
        except Exception as e:
            if is_enabled():
                dbg("eng.bath").debug(f"ERROR during sim.step: {type(e).__name__}: {e}")
            # Return a failure tuple to trigger controller retries/halving
            metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=1e9, mass_err=1e9)
            return False, metrics

        # Compute metrics conservatively
        try:
            if getattr(self.sim, "N", 0) > 0:
                vnorm = np.linalg.norm(self.sim.v, axis=1)
                vmax = float(np.max(vnorm))
                vmin = float(np.min(vnorm))
            else:
                vmax = 0.0
                vmin = 0.0
        except Exception:
            vmax = 0.0; vmin = 0.0
        try:
            m1 = float(np.sum(self.sim.m))
            mass_err = abs(m1 - m0) / max(m0, 1e-12)
        except Exception:
            mass_err = 0.0

        # Compute a stability hint for controller sidechain (do not enforce here)
        dt_hint = None
        try:
            if hasattr(self.sim, "_stable_dt"):
                dt_hint = float(self.sim._stable_dt())  # type: ignore[attr-defined]
        except Exception:
            dt_hint = None

        metrics = Metrics(
            max_vel=vmax,
            max_flux=vmax,
            div_inf=0.0,
            mass_err=mass_err,
            dt_limit=dt_hint,
        )
        self._last_metrics = metrics
        if is_enabled():
            dbg("eng.bath").debug(f"done: {pretty_metrics(metrics)} vmin={vmin:.3e}")
        return True, metrics

    def preferred_dt(self) -> Optional[float]:  # pragma: no cover - optional
        # Offer the simulator's stability hint if available
        dt0 = getattr(self.sim, "_stable_dt", None)
        try:
            return float(dt0()) if callable(dt0) else None
        except Exception:
            return None

    def get_metrics(self) -> Optional[Metrics]:  # pragma: no cover - optional
        return self._last_metrics


__all__ = ["BathDiscreteFluidEngine"]
