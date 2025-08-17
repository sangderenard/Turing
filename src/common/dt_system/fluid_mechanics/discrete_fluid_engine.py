# -*- coding: utf-8 -*-
"""DtCompatibleEngine adapter for Bath's DiscreteFluid (SPH).

Relocated from ``dt_system/bath_fluid_engine.py`` for consistency with
other fluid mechanics adapters. The class name and API are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine
from ..debug import dbg, is_enabled, pretty_metrics
if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..solids.api import SolidRegistry
from ..solids.api import GLOBAL_SOLIDS, GLOBAL_WORLD


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
    solids: Optional["SolidRegistry"] = None

    _last_metrics: Optional[Metrics] = None

    def __post_init__(self) -> None:
        # Default to global registry if none provided
        if self.solids is None:
            self.solids = GLOBAL_SOLIDS

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

        # Optional: pass solids registry to sim if supported
        try:
            if self.solids is not None and hasattr(self.sim, "set_solids"):
                self.sim.set_solids(self.solids)
        except Exception:
            pass

    # Advance exactly dt (sim may internally substep to remain stable)
        try:
            self.sim.step(float(dt))
        except Exception as e:
            if is_enabled():
                dbg("eng.bath").debug(f"ERROR during sim.step: {type(e).__name__}: {e}")
            # Return a failure tuple to trigger controller retries/halving
            metrics = Metrics(max_vel=0.0, max_flux=0.0, div_inf=1e9, mass_err=1e9)
            return False, metrics

    # Optional: apply world confinement for fluids after step (wrap/respawn)
        try:
            world = GLOBAL_WORLD
            # Per-plane override if any plane specifies a fluid_mode
            planes = getattr(world, "planes", []) or []
            plane_mode = None
            for pl in planes:
                pm = getattr(pl, "fluid_mode", None)
                if pm:
                    plane_mode = pm
                    break
            mode = plane_mode or getattr(world, "fluid_mode", None)
            bounds = getattr(world, "bounds", None)
            if mode and bounds is not None and hasattr(self.sim, "x"):
                import numpy as _np
                x = _np.asarray(self.sim.x)
                if x.ndim == 2 and x.shape[1] >= 2:
                    (mnx, mny, mnz), (mxx, mxy, mxz) = bounds
                    if mode == "wrap":
                        # modulo wrap within [min, max)
                        w = float(mxx - mnx)
                        h = float(mxy - mny)
                        if w > 0 and h > 0:
                            x[:, 0] = ((x[:, 0] - mnx) % w) + mnx
                            x[:, 1] = ((x[:, 1] - mny) % h) + mny
                    elif mode == "respawn":
                        # respawn out-of-bounds at random positions with avg speed direction
                        rng = _np.random.default_rng()
                        # Estimate average kinetic energy via velocity if available
                        if hasattr(self.sim, "v"):
                            v = _np.asarray(self.sim.v)
                            vnorm = _np.linalg.norm(v[:, :2], axis=1) if v.ndim == 2 else _np.zeros(x.shape[0])
                            v_avg = float(_np.mean(vnorm)) if vnorm.size else 0.0
                        else:
                            v_avg = 0.0
                        oob = (x[:, 0] < mnx) | (x[:, 0] > mxx) | (x[:, 1] < mny) | (x[:, 1] > mxy)
                        n_oob = int(_np.count_nonzero(oob))
                        if n_oob:
                            x[oob, 0] = rng.uniform(mnx, mxx, size=n_oob)
                            x[oob, 1] = rng.uniform(mny, mxy, size=n_oob)
                            if hasattr(self.sim, "v") and v_avg > 0:
                                theta = rng.uniform(0.0, 2.0 * _np.pi, size=n_oob)
                                self.sim.v[oob, 0] = v_avg * _np.cos(theta)
                                self.sim.v[oob, 1] = v_avg * _np.sin(theta)
        except Exception:
            pass

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
