# -*- coding: utf-8 -*-
"""DtCompatibleEngine adapter for Bath's DiscreteFluid (SPH).

Relocated from ``dt_system/bath_fluid_engine.py`` for consistency with
other fluid mechanics adapters. The class name and API are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple

import numpy as np

from ..dt_scaler import Metrics
from ..engine_api import DtCompatibleEngine
from ..debug import dbg, is_enabled, pretty_metrics
if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..solids.api import SolidRegistry
from ..solids.api import GLOBAL_SOLIDS, GLOBAL_WORLD
from ..solids.api import WorldConfinement
try:  # Local import for the bridge; keep optional to avoid import cycles when unused
    from src.cells.bath.discrete_fluid import DiscreteFluid, FluidParams  # type: ignore
except Exception:  # pragma: no cover - optional import
    DiscreteFluid = None  # type: ignore
    FluidParams = None  # type: ignore


@dataclass
class BathDiscreteFluidEngine(DtCompatibleEngine):
    # If using identity registration, dedup should default to False for fluids
    dedup: bool = False
    def register_vertices(self, state_table, positions, masses):
        uuids = []
        for pos, mass in zip(positions, masses):
            uuid_str = state_table.register_identity(pos, mass, dedup=self.dedup)
            uuids.append(uuid_str)
        return uuids
    def get_state(self, state=None):
        """
        Return the current state as a dict of relevant fields. If a state dict is supplied, update it in place.
        """
        out = state if isinstance(state, dict) else {}
        for k in ("x", "v", "m", "rho", "T", "S"):
            if hasattr(self.sim, k):
                out[k] = getattr(self.sim, k)
        return out
    """Adapter over :class:`DiscreteFluid` exposing DtCompatibleEngine.

    Parameters
    ----------
    sim:
        A DiscreteFluid instance (from ``src.cells.bath.discrete_fluid``).
    name:
        Optional display name (useful for logs/graphs).
    """

    # If sim is None and n is provided, a simple block will be created.
    sim: Optional[object] = None
    name: str = "bath.discrete"
    solids: Optional["SolidRegistry"] = None
    # Optional world descriptor; used to derive fluid bounds when auto-constructing
    world: Optional[WorldConfinement] = None

    # Optional convenience constructor params: build a block with ~n particles
    n: Optional[int] = None
    h: float = 0.12

    _last_metrics: Optional[Metrics] = None

    def __post_init__(self) -> None:
        # Default to global registry if none provided
        if self.solids is None:
            self.solids = GLOBAL_SOLIDS
        if self.world is None:
            self.world = GLOBAL_WORLD

        # Lazy-create a simple fluid block if requested via n and sim is not provided
        if self.sim is None and self.n is not None:
            if DiscreteFluid is None or FluidParams is None:
                raise RuntimeError("BathDiscreteFluidEngine bridge unavailable: bath DiscreteFluid not importable")
            n = max(1, int(self.n))
            # World box (inside which particles should start)
            wmin, wmax = _bounds_from_world(self.world)
            # Sim bounds: provide a margin outside the world so wrapping/respawn can happen without sim clamping
            margin = float(self.h) * 2.0
            sim_min = wmin - margin
            sim_max = wmax + margin

            # Choose a 2D block in XY with n_z=1; keep aspect near-square, but fit within world box minus inner padding
            dx = float(self.h) * 0.9
            pad = 2.0 * dx
            span = np.maximum(wmax - wmin - 2.0 * pad, 0.0)
            max_nx = max(1, int(np.floor(span[0] / max(dx, 1e-12))))
            max_ny = max(1, int(np.floor(span[1] / max(dx, 1e-12))))
            nx0 = max(1, int(np.ceil(np.sqrt(n))))
            ny0 = max(1, int(np.ceil(n / max(nx0, 1))))
            nx = min(max_nx, nx0)
            ny = min(max_ny, ny0)
            # Rebalance if ny got clamped too much
            if nx * ny < n:
                # Try to grow nx if possible
                grow = min(max_nx, int(np.ceil(n / max(ny, 1))))
                nx = max(1, grow)
                nx = min(nx, max_nx)
                ny = min(max_ny, int(np.ceil(n / max(nx, 1))))
            nz = 1

            xs = np.arange(nx) * dx
            ys = np.arange(ny) * dx
            zs = np.arange(nz) * dx
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            # Place the block fully inside the world box with padding on all sides
            origin = np.array([wmin[0] + pad, wmin[1] + pad, wmin[2] + pad], dtype=float)
            pos = pos + origin[None, :]
            # Trim to exactly n particles if we produced more
            if pos.shape[0] > n:
                pos = pos[:n, :]
            params = FluidParams(smoothing_length=self.h, particle_mass=0.02, bounce_damping=0.2)
            self.sim = DiscreteFluid(
                pos,
                velocities=None,
                temperature=None,
                salinity=None,
                params=params,
                bounds_min=(float(sim_min[0]), float(sim_min[1]), float(sim_min[2])),
                bounds_max=(float(sim_max[0]), float(sim_max[1]), float(sim_max[2])),
            )

    # -------- Bridge factories ------------------------------------------------
    @classmethod
    def from_engine(cls, sim: object, *, name: str = "bath.discrete", solids: Optional["SolidRegistry"] = None):
        """Wrap an existing DiscreteFluid engine.

        This keeps all construction outside, but centralizes the bridge point.
        """
        return cls(sim=sim, name=name, solids=solids)

    @classmethod
    def from_block(
        cls,
        *,
        n_x: int = 20,
        n_y: int = 24,
        n_z: int = 1,
        h: float = 0.12,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(2.0, 2.0, 2.0),
        name: str = "bath.discrete",
        solids: Optional["SolidRegistry"] = None,
    world: Optional[WorldConfinement] = None,
    ):
        """Construct a simple rectangular block DiscreteFluid and wrap it.

        Intended for demos/tools that want a one-liner bridge without importing
        bath modules at call sites. This method imports DiscreteFluid lazily.
        """
        if DiscreteFluid is None or FluidParams is None:
            raise RuntimeError("BathDiscreteFluidEngine bridge unavailable: bath DiscreteFluid not importable")
        dx = h * 0.9
        xs = np.arange(n_x) * dx
        ys = np.arange(n_y) * dx
        zs = np.arange(n_z) * dx
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        # If a world is provided, place the block inside the world box with an inner padding
        if world is not None:
            wmin, wmax = _bounds_from_world(world)
            pad = 2.0 * dx
            origin = np.array([wmin[0] + pad, wmin[1] + pad, wmin[2] + pad], dtype=float)
            pos = pos + origin[None, :]
        else:
            # Legacy default: lift above ground a bit
            pos[:, 1] += 0.2
        params = FluidParams(smoothing_length=h, particle_mass=0.02, bounce_damping=0.2)
        # Expand sim bounds beyond world box if provided so higher-level wrap/respawn can manage boundaries
        if world is not None:
            wmin, wmax = _bounds_from_world(world)
            margin = 2.0 * h
            bmin_eff = (float(wmin[0] - margin), float(wmin[1] - margin), float(wmin[2] - margin))
            bmax_eff = (float(wmax[0] + margin), float(wmax[1] + margin), float(wmax[2] + margin))
        else:
            bmin_eff = bounds_min
            bmax_eff = bounds_max
        sim = DiscreteFluid(
            pos,
            velocities=None,
            temperature=None,
            salinity=None,
            params=params,
            bounds_min=bmin_eff,
            bounds_max=bmax_eff,
        )
        return cls(sim=sim, name=name, solids=solids, world=world)

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

    def step(self, dt: float, state=None, state_table=None):
        # If state is provided, update sim fields from it
        if isinstance(state, dict):
            for k in ("x", "v", "m", "rho", "T", "S"):
                if k in state and hasattr(self.sim, k):
                    setattr(self.sim, k, state[k])
        # Optionally use state_table for advanced cross-system metrics (not implemented here)
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
        return True, metrics, self.get_state()


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


# --------------------- helpers ---------------------------------------------

def _bounds_from_world(world: Optional[WorldConfinement]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an axis-aligned fluid bounds box from world planes.

    Priority: explicit world.bounds -> derived from axis-aligned planes -> fallback defaults.
    Plane convention: n·x + d >= 0 is inside. For axis-aligned planes:
      +X: x >= -d,  -X: x <= d; similarly for Y/Z.
    """
    # Default fallback (legacy demo values)
    default_min = np.array([0.0, 0.0, 0.0], dtype=float)
    default_max = np.array([2.0, 2.0, 2.0], dtype=float)

    if world is None:
        return default_min, default_max

    mn = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    mx = np.array([+np.inf, +np.inf, +np.inf], dtype=float)

    try:
        planes = list(getattr(world, "planes", []) or [])
    except Exception:
        planes = []

    def _is_axis(vec: np.ndarray, axis: int, sgn: int, tol: float = 1e-6) -> bool:
        # sgn = +1 for +axis unit, -1 for -axis unit
        t = np.zeros(3, dtype=float)
        t[axis] = float(sgn)
        return np.linalg.norm(vec - t, ord=2) <= tol

    for pl in planes:
        try:
            n = np.asarray(getattr(pl, "normal", np.array([0.0, 1.0, 0.0], dtype=float)), dtype=float).reshape(3)
            d = float(getattr(pl, "offset", 0.0))
        except Exception:
            continue
        # Only consider planes whose half-space contains the origin (world center)
        # i.e., n·0 + d >= 0 => d >= 0
        if d < 0.0:
            continue
        # X
        if _is_axis(n, 0, +1):
            mn[0] = max(mn[0], -d)
        elif _is_axis(n, 0, -1):
            mx[0] = min(mx[0], d)
        # Y
        elif _is_axis(n, 1, +1):
            mn[1] = max(mn[1], -d)
        elif _is_axis(n, 1, -1):
            mx[1] = min(mx[1], d)
        # Z
        elif _is_axis(n, 2, +1):
            mn[2] = max(mn[2], -d)
        elif _is_axis(n, 2, -1):
            mx[2] = min(mx[2], d)

    # If any components remain infinite, try world.bounds as a fallback for those axes
    try:
        if world.bounds is not None:
            (bmnx, bmny, bmnz), (bmxx, bmxy, bmxz) = world.bounds  # type: ignore[assignment]
            wb_min = np.array([float(bmnx), float(bmny), float(bmnz)], dtype=float)
            wb_max = np.array([float(bmxx), float(bmxy), float(bmxz)], dtype=float)
        else:
            wb_min = default_min; wb_max = default_max
    except Exception:
        wb_min = default_min; wb_max = default_max

    # Resolve infinities with world bounds, then defaults
    out_min = np.where(np.isfinite(mn), mn, wb_min)
    out_max = np.where(np.isfinite(mx), mx, wb_max)

    # Ensure min <= max component-wise
    out_min = np.minimum(out_min, out_max)
    out_max = np.maximum(out_max, out_min)
    return out_min.astype(float), out_max.astype(float)
