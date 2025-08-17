from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..data.state import Cell, Bath
from .provider import MechanicsProvider, MechanicsSnapshot

# Softbody engine imports (kept optional to avoid import cost if unused)
from src.cells.softbody.engine.params import EngineParams
from src.cells.softbody.engine.hierarchy import Hierarchy, build_cell
from src.cells.softbody.engine.xpbd_core import XPBDSolver
from src.cells.softbody.engine.coupling import (
    harmonized_update,
    laplace_from_Ka,
    cell_area,
)
from src.cells.softbody.engine.fields import FieldStack
from src.common.dt_system.dt_controller import STController, Targets


@dataclass
class SoftbodyProviderCfg:
    substeps: int = 20
    dt_provider: float = 1.0 / 60.0
    pressure_scale: float = 1.0
    area_scale: float = 1.0
    # Radius range for placement in provider units (box [0,1]^2)
    r_min: float = 0.04
    r_max: float = 0.10
    dim: int = 3


class Softbody0DProvider(MechanicsProvider):
    """XPBD softbody-backed 0D mechanics provider.

    Builds a tiny softbody world and advances it each step; publishes per-cell
    contact pressure and surface area as 0D aggregates for the cellsim engine.
    """

    def __init__(self, cfg: Optional[SoftbodyProviderCfg] = None):
        self.cfg = cfg or SoftbodyProviderCfg()
        self._h: Optional[Hierarchy] = None
        self._params: Optional[EngineParams] = None
        self._ids: List[str] = []
        self._cells: Optional[List[Cell]] = None
        self._bath: Optional[Bath] = None
        self.dt_ctrl = STController()
        self.dt_targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
        self.dx = 1.0

    # MechanicsProvider -----------------------------------------------------
    def sync(self, cells: List[Cell], bath: Bath) -> None:
        self._cells = cells
        self._bath = bath
        # (Re)build world if needed or cell count changed
        if self._h is None or len(self._ids) != len(cells):
            self._build_world(cells)

        # Map cell state onto provider controls using vectorized operations
        assert self._h is not None
        V = np.maximum(1e-18, np.asarray([getattr(c, "V", 0.0) for c in cells], dtype=float))
        elastic_k = np.asarray([getattr(c, "elastic_k", 0.0) for c in cells], dtype=float)
        imp = np.asarray([getattr(c, "n", {}).get("Imp", 0.0) for c in cells], dtype=float)
        P_ext = float(getattr(bath, "pressure", 0.0))

        V_min, V_max = float(np.min(V)), float(np.max(V))
        if V_max <= V_min:
            if self.cfg.dim == 2:
                mapped_vols = np.full_like(V, math.pi * (self.cfg.r_min ** 2))
            elif self.cfg.dim == 1:
                mapped_vols = np.full_like(V, 2.0 * self.cfg.r_min)
            else:
                mapped_vols = np.full_like(V, (4.0 / 3.0) * math.pi * (self.cfg.r_min ** 3))
        else:
            alpha = (V - V_min) / max(1e-18, (V_max - V_min))
            r = self.cfg.r_min * (1 - alpha) + self.cfg.r_max * alpha
            if self.cfg.dim == 2:
                mapped_vols = math.pi * (r ** 2)
            elif self.cfg.dim == 1:
                mapped_vols = 2.0 * r
            else:
                mapped_vols = (4.0 / 3.0) * math.pi * (r ** 3)
        osmotic = imp / np.maximum(V, 1e-18)

        for sbc, V_t, k, osm in zip(self._h.cells, mapped_vols, elastic_k, osmotic):
            sbc.constraints["volume"].target = V_t
            sbc.membrane_tension = k
            sbc.osmotic_pressure = osm
            sbc.external_pressure = P_ext

    # Array-first sync path to avoid Python conversions completely
    def sync_arrays(self, *, V, elastic_k, imp, bath_pressure: float, bath_temperature: float) -> None:  # type: ignore[override]
        if self._h is None:
            return
        V = np.maximum(1e-18, np.asarray(V, dtype=float))
        elastic_k = np.asarray(elastic_k, dtype=float)
        imp = np.asarray(imp, dtype=float)
        n = V.shape[0]
        # Map volumes to provider target volumes
        V_min = float(np.min(V))
        V_max = float(np.max(V))
        if V_max <= V_min:
            if self.cfg.dim == 2:
                mapped_vols = np.full(n, math.pi * (self.cfg.r_min ** 2), dtype=float)
            elif self.cfg.dim == 1:
                mapped_vols = np.full(n, 2.0 * self.cfg.r_min, dtype=float)
            else:
                mapped_vols = np.full(n, (4.0 / 3.0) * math.pi * (self.cfg.r_min ** 3), dtype=float)
        else:
            alpha = (V - V_min) / max(1e-18, (V_max - V_min))
            r = self.cfg.r_min * (1 - alpha) + self.cfg.r_max * alpha
            if self.cfg.dim == 2:
                mapped_vols = math.pi * (r ** 2)
            elif self.cfg.dim == 1:
                mapped_vols = 2.0 * r
            else:
                mapped_vols = (4.0 / 3.0) * math.pi * (r ** 3)
        osmotic = imp / np.maximum(V, 1e-18)
        for sbc, V_t, k, osm in zip(self._h.cells, mapped_vols, elastic_k, osmotic):
            sbc.constraints["volume"].target = V_t
            sbc.membrane_tension = k
            sbc.osmotic_pressure = osm
            sbc.external_pressure = float(bath_pressure)

        # Cache bath conditions for step phase if needed
        self._bath_pressure = float(bath_pressure)
        self._bath_temperature = float(bath_temperature)

    def step(self, dt: float, *, hooks=None) -> MechanicsSnapshot:
        from src.common.sim_hooks import SimHooks

        hooks = hooks or SimHooks()
        if self._h is None or self._cells is None or self._bath is None:
            return {}

        # Substep sizing: prefer caller's dt, fall back to provider default
        nsub = max(1, int(self.cfg.substeps))
        base_dt = float(dt) if (dt is not None and dt > 0.0) else float(self.cfg.dt_provider)
        sub_dt = base_dt / nsub
        dt_curr = sub_dt

        h = self._h

        # Batch views (no copies of vertex data)
        cells_arr = np.array(h.cells, dtype=object)
        Ka_arr = np.array([float(c.elastic_k) for c in self._cells], dtype=float)
        Kb_arr = np.array([float(getattr(c, "bulk_modulus", 1e5)) for c in self._cells], dtype=float)
        P_osm_arr = np.array([float(getattr(sbc, "osmotic_pressure", 0.0)) for sbc in cells_arr], dtype=float)
        P_ext = float(getattr(self._bath, "pressure", 0.0))
        P_ext_arr = np.full(len(cells_arr), P_ext, dtype=float)

        # 0D → softbody parameter harmonization (batched over arrays using a tight Python loop).
        # np.vectorize doesn't offer speedups; keep scalar function but drive with arrays.
        def _harmonized_update_batch():
            for sbc, ka, kb, po, pe in zip(cells_arr, Ka_arr, Kb_arr, P_osm_arr, P_ext_arr):
                harmonized_update(sbc, ka, kb, po, pe)

        # World time (lazy-init)
        t_now = getattr(self, "_t", 0.0)

        for _ in range(nsub):
            _harmonized_update_batch()

            hooks.run_pre(self, dt_curr)

            # Single canonical softbody step with adaptive dt
            _, dt_curr = h.step_dt_control(
                dt_curr,
                self.dt_ctrl,
                self.dt_targets,
                dx=self.dx,
                t=t_now,
                fields=getattr(h, "fields", None),
            )

            # Organelle mode updates (your existing hook)
            h.update_organelle_modes(dt_curr)

            hooks.run_post(self, dt_curr)

            # Advance provider time
            t_now += dt_curr

        # Persist time back to provider
        self._t = t_now

        # Observables (vectorized)
        V = np.fromiter((abs(float(sbc.enclosed_volume())) for sbc in cells_arr), count=len(cells_arr), dtype=float)
        A = np.fromiter((cell_area(sbc) for sbc in cells_arr), count=len(cells_arr), dtype=float)
        laplace = np.array([laplace_from_Ka(sbc, ka) for sbc, ka in zip(cells_arr, Ka_arr)], dtype=float)
        gamma = laplace[:, 0]
        dP_L = laplace[:, 1]
        P_internal = P_ext + dP_L

        # Scaled outputs
        pressures = self.cfg.pressure_scale * P_internal
        areas     = self.cfg.area_scale * A
        vols      = V

        # Write back aggregates to 0D cells
        for c0d, g, P_i, vol in zip(self._cells or [], gamma, P_internal, V):
            # Keep as raw numpy-friendly types; avoid forcing to Python floats
            c0d.membrane_tension = g
            c0d.internal_pressure = P_i
            c0d.measured_volume = vol

        return {"pressures": pressures, "areas": areas, "volumes": vols}

    # Internals -------------------------------------------------------------
    def _build_world(self, cells: List[Cell]) -> None:
        n = len(cells)
        self._params = EngineParams(dimension=self.cfg.dim)
        if self.cfg.dim == 1:
            self._params.bath_min = (-1.0, 0.0, -1.0)
            self._params.bath_max = (1.0, 0.0, -1.0)
        elif self.cfg.dim == 2:
            self._params.bath_min = (-1.0, -1.0, -1.0)
            self._params.bath_max = (1.0, 1.0, -1.0)
        else:
            self._params.bath_min = (-1.0, -1.0, -1.0)
            self._params.bath_max = (1.0, 1.0, 1.00)
        solver = XPBDSolver(self._params)

        # Simple layout: grid in [0.2,0.8], collapse unused dims
        if self.cfg.dim == 1:
            grid_cols = n
            grid_rows = 1
            xs = np.linspace(0.2, 0.8, grid_cols)
            ys = np.full(1, 0.5)
        else:
            grid_cols = max(1, int(math.ceil(math.sqrt(n))))
            grid_rows = max(1, int(math.ceil(n / grid_cols)))
            xs = np.linspace(0.2, 0.8, grid_cols)
            ys = np.linspace(0.2, 0.8, grid_rows)

        # Radii mapped from current V set
        V_list = [max(1e-18, float(c.V)) for c in cells]
        V_min, V_max = min(V_list), max(V_list)
        def V_to_r(V: float) -> float:
            if V_max <= V_min:
                return self.cfg.r_min
            alpha = (V - V_min) / max(1e-18, (V_max - V_min))
            return self.cfg.r_min * (1 - alpha) + self.cfg.r_max * alpha

        sb_cells = []
        self._ids = []
        idx = 0
        for j in range(grid_rows):
            for i in range(grid_cols):
                if idx >= n:
                    break
                cx, cy = float(xs[i]), float(ys[j])
                r = V_to_r(V_list[idx])
                if self.cfg.dim == 2:
                    target = math.pi * r ** 2
                    center = (cx, cy, 0.0)
                elif self.cfg.dim == 1:
                    target = 2.0 * r
                    center = (cx, 0.5, 0.0)
                else:
                    target = (4.0 / 3.0) * math.pi * r ** 3
                    center = (cx, cy, 0.01)
                X, F, V, invm, edges, bends, constraints = build_cell(
                    id_str=f"cell{idx}", center=center, radius=r,
                    params=self._params, subdiv=1, mass_per_vertex=1.0, target_volume=target
                )
                # Build minimal shim that Hierarchy expects
                from src.cells.softbody.engine.hierarchy import Cell as SBC
                sb_cell = SBC(
                    id=f"cell{idx}", X=X, V=V, invm=invm, faces=F, edges=edges,
                    bends=bends, constraints=constraints, organelles=[], dim=self.cfg.dim,
                )
                sb_cell.membrane_tension = float(cells[idx].elastic_k)
                sb_cell.osmotic_pressure = 0.0
                sb_cells.append(sb_cell)
                self._ids.append(sb_cell.id)
                idx += 1

        self._h = Hierarchy(
            box_min=np.array(self._params.bath_min, dtype=float),
            box_max=np.array(self._params.bath_max, dtype=float),
            cells=sb_cells,
            solver=solver,
            params=self._params,
        )
        # Attach a field stack for optional environmental forces/noise
        self._h.fields = FieldStack()
        # Optional: attach default fields (noise, gravity demo). Safe if library is available.
        try:
            from src.cells.softbody.resources.field_library import (
                uniform_flow,
                shear_flow,
                fluid_noise,
                gravity,
            )
            # gentle Brownian jiggle (no COM drift)
            self._h.fields.add(fluid_noise(sigma=.5e-4, com_neutral=True, dim=self.cfg.dim))
            # example gravity on a subset
            self._h.fields.add(
                gravity(
                    g=(0, -0.4, 0) if self.cfg.dim == 3 else (0, -0.4, 0),
                    selector=lambda c: c.id in {"cell0", "cell1", "cell2"},
                    dim=self.cfg.dim,
                )
            )
        except Exception:
            # Field library optional; proceed without if unavailable
            pass
