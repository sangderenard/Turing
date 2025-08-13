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

        # Map cell state onto provider controls
        assert self._h is not None
        V_list = [max(1e-18, float(c.V)) for c in cells]
        V_min, V_max = min(V_list), max(V_list)
        # Normalize target volumes to a reasonable range in provider units
        def map_volume(V: float) -> float:
            if V_max <= V_min:
                return (4.0 / 3.0) * math.pi * (self.cfg.r_min ** 3)
            alpha = (V - V_min) / max(1e-18, (V_max - V_min))
            r = self.cfg.r_min * (1 - alpha) + self.cfg.r_max * alpha
            return (4.0 / 3.0) * math.pi * (r ** 3)

        for i, (c, sbc) in enumerate(zip(cells, self._h.cells)):
            sbc.constraints["volume"].target = map_volume(float(c.V))
            sbc.membrane_tension = float(c.elastic_k)
            ci_imp = 0.0
            if "Imp" in c.n:
                ci_imp = float(c.n.get("Imp", 0.0) / max(c.V, 1e-18))
            sbc.osmotic_pressure = ci_imp
            sbc.external_pressure = float(getattr(bath, "pressure", 0.0))

    def step(self, dt: float) -> MechanicsSnapshot:
        if self._h is None or self._cells is None or self._bath is None:
            return {}
        p = self._h.params
        sub_dt = self.cfg.dt_provider / max(1, int(self.cfg.substeps))
        for _ in range(self.cfg.substeps):
            for c0d, sbc in zip(self._cells, self._h.cells):
                harmonized_update(
                    sbc,
                    float(c0d.elastic_k),
                    float(getattr(c0d, "bulk_modulus", 1e5)),
                    float(getattr(sbc, "osmotic_pressure", 0.0)),
                    float(getattr(self._bath, "pressure", 0.0)),
                )
            self._h.integrate(sub_dt)
            self._h.project_constraints(sub_dt)
            self._h.update_organelle_modes(sub_dt)

        pressures: List[float] = []
        areas: List[float] = []
        vols: List[float] = []
        for c0d, sbc in zip(self._cells, self._h.cells):
            V = abs(float(sbc.enclosed_volume()))
            A = cell_area(sbc)
            gamma, dP_L = laplace_from_Ka(sbc, float(c0d.elastic_k))
            P_internal = float(getattr(self._bath, "pressure", 0.0)) + dP_L
            pressures.append(self.cfg.pressure_scale * P_internal)
            areas.append(self.cfg.area_scale * A)
            vols.append(V)
            c0d.membrane_tension = gamma
            c0d.internal_pressure = P_internal
            c0d.measured_volume = V
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
