from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..data.state import Cell, Bath
from .provider import MechanicsProvider, MechanicsSnapshot

# Softbody engine imports (kept optional to avoid import cost if unused)
from src.transmogrifier.softbody.engine.params import EngineParams
from src.transmogrifier.softbody.engine.hierarchy import Hierarchy, build_cell
from src.transmogrifier.softbody.engine.xpbd_core import XPBDSolver
from src.transmogrifier.softbody.engine.coupling import (
    harmonized_update,
    laplace_from_Ka,
    cell_area,
)


@dataclass
class SoftbodyProviderCfg:
    substeps: int = 4
    dt_provider: float = 1.0 / 60.0
    pressure_scale: float = 1.0
    area_scale: float = 1.0
    # Radius range for placement in provider units (box [0,1]^2)
    r_min: float = 0.04
    r_max: float = 0.10


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

    # MechanicsProvider -----------------------------------------------------
    def sync(self, cells: List[Cell], bath: Bath) -> None:
        self._cells = cells
        self._bath = bath
        # (Re)build world if needed or cell count changed
        if self._h is None or len(self._ids) != len(cells):
            self._build_world(cells)

        # Map cell state onto provider controls using vectorized operations
        assert self._h is not None
        V = np.maximum(1e-18, np.array([float(c.V) for c in cells], dtype=float))
        elastic_k = np.array([float(c.elastic_k) for c in cells], dtype=float)
        imp = np.array([float(c.n.get("Imp", 0.0)) for c in cells], dtype=float)
        P_ext = float(getattr(bath, "pressure", 0.0))

        V_min, V_max = float(np.min(V)), float(np.max(V))
        if V_max <= V_min:
            mapped_vols = np.full_like(
                V, (4.0 / 3.0) * math.pi * (self.cfg.r_min ** 3)
            )
        else:
            alpha = (V - V_min) / max(1e-18, (V_max - V_min))
            r = self.cfg.r_min * (1 - alpha) + self.cfg.r_max * alpha
            mapped_vols = (4.0 / 3.0) * math.pi * (r ** 3)
        osmotic = imp / np.maximum(V, 1e-18)

        for sbc, V_t, k, osm in zip(self._h.cells, mapped_vols, elastic_k, osmotic):
            sbc.constraints["volume"].target = float(V_t)
            sbc.membrane_tension = float(k)
            sbc.osmotic_pressure = float(osm)
            sbc.external_pressure = P_ext

    def step(self, dt: float) -> MechanicsSnapshot:
        if self._h is None or self._cells is None or self._bath is None:
            return {}

        # Substep sizing: prefer caller's dt, fall back to provider default
        nsub = max(1, int(self.cfg.substeps))
        base_dt = float(dt) if (dt is not None and dt > 0.0) else float(self.cfg.dt_provider)
        sub_dt = base_dt / nsub

        h = self._h

        # Batch views (no copies of vertex data)
        cells_arr = np.array(h.cells, dtype=object)
        Ka_arr = np.array([float(c.elastic_k) for c in self._cells], dtype=float)
        Kb_arr = np.array([float(getattr(c, "bulk_modulus", 1e5)) for c in self._cells], dtype=float)
        P_osm_arr = np.array([float(getattr(sbc, "osmotic_pressure", 0.0)) for sbc in cells_arr], dtype=float)
        P_ext = float(getattr(self._bath, "pressure", 0.0))
        P_ext_arr = np.full(len(cells_arr), P_ext, dtype=float)

        # 0D → softbody parameter harmonization (vectorized)
        def _harmonized_update_batch():
            vec = np.vectorize(
                lambda sbc, ka, kb, po, pe: harmonized_update(sbc, ka, kb, po, pe),
                otypes=[object],
            )
            vec(cells_arr, Ka_arr, Kb_arr, P_osm_arr, P_ext_arr)

        # World time (lazy-init)
        t_now = getattr(self, "_t", 0.0)

        for _ in range(nsub):
            _harmonized_update_batch()

            # Single canonical softbody step (predict → fields → project → rebuild V)
            # Pass fields stack if you've attached one on the hierarchy (optional).
            h.step(sub_dt, t=t_now, fields=getattr(h, "fields", None))

            # Organelle mode updates (your existing hook)
            h.update_organelle_modes(sub_dt)

            # Advance provider time
            t_now += sub_dt

        # Persist time back to provider
        self._t = t_now

        # Observables (vectorized)
        V = np.array([abs(float(sbc.enclosed_volume())) for sbc in cells_arr], dtype=float)
        A = np.array([cell_area(sbc) for sbc in cells_arr], dtype=float)
        laplace = np.array([laplace_from_Ka(sbc, ka) for sbc, ka in zip(cells_arr, Ka_arr)], dtype=float)
        gamma = laplace[:, 0]
        dP_L = laplace[:, 1]
        P_internal = P_ext + dP_L

        # Scaled outputs
        pressures = self.cfg.pressure_scale * P_internal
        areas     = self.cfg.area_scale * A
        vols      = V

        # Write back aggregates to 0D cells
        for c0d, g, P_i, vol in zip(self._cells, gamma, P_internal, V):
            c0d.membrane_tension = float(g)
            c0d.internal_pressure = float(P_i)
            c0d.measured_volume = float(vol)

        return {"pressures": pressures, "areas": areas, "volumes": vols}

    # Internals -------------------------------------------------------------
    def _build_world(self, cells: List[Cell]) -> None:
        n = len(cells)
        self._params = EngineParams()
        solver = XPBDSolver(self._params)

        # Simple layout: grid in [0.2,0.8]^2, z=0.01
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
                X, F, V, invm, edges, bends, constraints = build_cell(
                    id_str=f"cell{idx}", center=(cx, cy, 0.01), radius=r,
                    params=self._params, subdiv=1, mass_per_vertex=1.0, target_volume=(4.0/3.0)*math.pi*r**3
                )
                # Build minimal shim that Hierarchy expects
                from src.transmogrifier.softbody.engine.hierarchy import Cell as SBC
                sb_cell = SBC(
                    id=f"cell{idx}", X=X, V=V, invm=invm, faces=F, edges=edges,
                    bends=bends, constraints=constraints, organelles=[],
                )
                sb_cell.membrane_tension = float(cells[idx].elastic_k)
                sb_cell.osmotic_pressure = 0.0
                sb_cells.append(sb_cell)
                self._ids.append(sb_cell.id)
                idx += 1

        self._h = Hierarchy(
            box_min=np.array([0.0, 0.0, 0.0]),
            box_max=np.array([1.0, 1.0, 0.02]),
            cells=sb_cells,
            solver=solver,
            params=self._params,
        )
        # after self._h = Hierarchy(...)
        from src.transmogrifier.softbody.resources.field_library import uniform_flow, shear_flow, fluid_noise, gravity

        # whole-system drift to +X
        self._h.fields.add(uniform_flow(u=(0.03, 0.0, 0.0), dim=3))

        # visible shear (u_x = rate * y)
        # self._h.fields.add(shear_flow(rate=0.5, axis_xy=(0,1), dim=3))

        # gentle Brownian jiggle (no COM drift)
        #self._h.fields.add(fluid_noise(sigma=5e-4, com_neutral=True, dim=3))

        # gravity on a subset (example: only cell0 & cell2)
        # self._h.fields.add(gravity(g=(0,-0.4,0), selector=lambda c: c.id in {"cell0","cell2"}, dim=3))
