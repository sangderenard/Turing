# cellsim/api/saline.py
from __future__ import annotations
import math
from dataclasses import dataclass
import logging
import json
import sys
from typing import Iterable, List, Sequence, Optional
from sympy import lambdify, symbols, Integer, Float
import numpy as np
from ..engine.saline import SalineEngine
from ..data.state import Cell, Bath, Organelle
from ..core.geometry import sphere_area_from_volume
from tqdm.auto import tqdm  # type: ignore

# Module logger (DEBUG by default so logs appear without extra setup)
logger = logging.getLogger("cellsim.api.saline")
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stdout)
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)

@dataclass
class IntegerAllocatorCfg:
    method: str = "adams"          # "truncate" | "adams"
    protect_under_one: bool = True
    bump_under_one: bool = True

class SalinePressureAPI:
    """
    Back-compat facade:
      - Keeps your s/p-expression bar view + integer allocator.
      - Builds cellsim Cell/Bath and runs SalineEngine for physics.
      - Optional hooks to BitBitBuffer for expand/snap.
    """
    def __init__(self,
                 cells: List[Cell],
                 bath: Bath,
                 *,
                 species: Iterable[str] = ("Na","K","Cl","Imp"),
                 epsilon: float = 1e-6,
                 int_alloc: IntegerAllocatorCfg = IntegerAllocatorCfg(),
                 chars: Optional[List[str]] = None,
                 width: int = 80,
                 s_exprs: Optional[Sequence] = None,
                 p_exprs: Optional[Sequence] = None):
        self.cells = cells
        self.bath = bath
        self.species = tuple(species)
        self.epsilon = float(epsilon)
        self.int_alloc = int_alloc
        self.chars = chars or [chr(97+i) for i in range(len(cells))]
        self.width = int(width)

        # optional bar-driver (legacy “view”)
        self.t = symbols('t')
        self.s_exprs = list(s_exprs) if s_exprs is not None else [Integer(c.n.get("Imp", 0.0)) for c in cells]
        self.p_exprs = list(p_exprs) if p_exprs is not None else [Float(getattr(c, "base_pressure", 0.0)) for c in cells]
        self.s_funcs = [lambdify(self.t, e, 'math') for e in self.s_exprs]
        self.p_funcs = [lambdify(self.t, e, 'math') for e in self.p_exprs]

        # init engine
        for c in tqdm(self.cells, desc="cells", leave=False):
            if c.A0 <= 0.0:
                A0, _ = sphere_area_from_volume(c.V); c.A0 = A0
        self.engine = SalineEngine(self.cells, self.bath, species=self.species)

    # ---- debug helpers ----
    def _compartment_snapshot(self, comp, species: List[str]):
        V = float(getattr(comp, "V", 0.0))
        n = {sp: float(comp.n.get(sp, 0.0)) for sp in species}
        conc = getattr(comp, "conc", None)
        concs = conc(species) if callable(conc) else {sp: (n.get(sp, 0.0)/max(V, 1e-18)) for sp in species}
        snap = {
            "V": V,
            "n": n,
            "conc": {sp: float(concs.get(sp, 0.0)) for sp in species},
        }
        # Bath extras
        for extra in ("pressure", "temperature", "compressibility"):
            if hasattr(comp, extra):
                snap[extra] = float(getattr(comp, extra))
        # Cell extras
        for extra in ("A0", "elastic_k", "visc_eta", "Lp0", "base_pressure"):
            if hasattr(comp, extra):
                snap[extra] = float(getattr(comp, extra))
        # Mechanics (may be populated by engine)
        if hasattr(comp, "pressure"):
            try:
                snap["pressure"] = float(getattr(comp, "pressure"))
            except Exception:
                pass
        return snap

    def snapshot_system(self, legacy_sim) -> dict:
        species = list(self.species)
        # Bath
        bath_snap = self._compartment_snapshot(self.bath, species)
        # Cells
        cells = []
        for i, (leg, cs) in enumerate(zip(getattr(legacy_sim, "cells", []), self.cells)):
            cell_snap = self._compartment_snapshot(cs, species)
            # label/geometry from legacy for traceability
            for k in ("label", "left", "right", "leftmost", "rightmost"):
                if hasattr(leg, k):
                    try:
                        cell_snap[k] = getattr(leg, k)
                    except Exception:
                        pass
            # organelles
            organelles = []
            for o in getattr(cs, "organelles", []) or []:
                o_snap = {
                    "volume_total": float(getattr(o, "volume_total", 0.0)),
                    "lumen_fraction": float(getattr(o, "lumen_fraction", 0.0)),
                    "V_lumen": float(o.V if hasattr(o, "V") else 0.0),
                    "V_solid": float(getattr(o, "V_solid", 0.0)),
                    "incompressible": bool(getattr(o, "incompressible", False)),
                    "n": {sp: float(o.n.get(sp, 0.0)) for sp in species},
                    "conc": {sp: float((o.n.get(sp, 0.0) / max(o.V, 1e-18))) for sp in species},
                }
                organelles.append(o_snap)
            if organelles:
                cell_snap["organelles"] = organelles
            cells.append(cell_snap)
        return {
            "species": species,
            "bath": bath_snap,
            "cells": cells,
        }

    # ---- legacy “view”: equilibrium fractions & bar ----
    def equilibrium_fracs(self, t: float) -> List[float]:
        s_vals = [f(t) for f in self.s_funcs]
        p_vals = [f(t) for f in self.p_funcs]
        r = []
        eps = self.epsilon
        for si, pi in tqdm(list(zip(s_vals, p_vals)), desc="exprs", leave=False):
            denom = pi if abs(pi) > eps else eps
            r.append(si/denom)
        s = sum(r); 
        if abs(s) < eps:
            return [1.0/len(r)]*len(r)
        return [ri/s for ri in r]

    def equilibrium_bar(self, t: float) -> str:
        fracs = self.equilibrium_fracs(t)
        if self.int_alloc.method == "truncate":
            segs = self._integer_allocate_truncate(fracs, self.width)
        else:
            segs = self._integer_allocate_adams(fracs, self.width)
        return '|' + ''.join(self.chars[i]*segs[i] for i in range(len(segs))) + '|'

    def _integer_allocate_truncate(self, fracs, W):
        from math import floor, ceil
        quotas = [f*W for f in fracs]
        ceilings = [ceil(q) for q in quotas]
        K = sum(ceilings) - W
        if K <= 0: return ceilings
        costs = []
        for i, q in tqdm(list(enumerate(quotas)), desc="quotas", leave=False):
            if q <= 1.0 and self.int_alloc.protect_under_one:
                cost = float('inf')
            else:
                cost = 1 - (q - floor(q))
            costs.append((cost, i))
        for _, idx in tqdm(sorted(costs, key=lambda x: x[0])[:K], desc="ceilings", leave=False):
            ceilings[idx] -= 1
        return ceilings

    def _integer_allocate_adams(self, fracs, W):
        from math import floor, ceil
        quotas = [f*W for f in fracs]
        ceilings = [max(1, ceil(q)) if self.int_alloc.bump_under_one else ceil(q) for q in quotas]
        K = sum(ceilings) - W
        if K <= 0: return ceilings
        costs = []
        for i, q in tqdm(list(enumerate(quotas)), desc="quotas", leave=False):
            if q < 1.0 and self.int_alloc.protect_under_one:
                cost = float('inf')
            else:
                cost = 1 - (q - floor(q))
            costs.append((cost, i))
        for _, idx in tqdm(sorted(costs, key=lambda x: x[0])[:K], desc="ceilings", leave=False):
            ceilings[idx] -= 1
        return ceilings

    # ---- physics step (cellsim backend) ----
    def step(self, dt: float) -> float:
        """Returns suggested next dt (adaptive)."""
        return self.engine.step(dt)

    # ---- helpers to construct from your legacy sim object ----
    @classmethod
    def from_legacy(cls, sim) -> "SalinePressureAPI":
        # Map legacy cells → Cell; salinity → Imp; pressure → base_pressure
        cells = []
        for legacy in tqdm(sim.cells, desc="cells", leave=False):
            V = float(legacy.right - legacy.left)
            cell = Cell(V=V,
                        n={"Imp": float(getattr(legacy, "salinity", 0.0)),
                           "Na": 0.0, "K": 0.0, "Cl": 0.0},
                        base_pressure=float(getattr(legacy, "pressure", 0.0)),
                        elastic_k=float(getattr(legacy, "elastic_k", 0.1)),
                        visc_eta=float(getattr(legacy, "visc_eta", 0.0)))
            # organelles if present
            if hasattr(legacy, "organelles"):
                for o in tqdm(legacy.organelles, desc="organelles", leave=False):
                    cell.organelles.append(
                        Organelle(volume_total=float(o.volume_total),
                                  lumen_fraction=float(getattr(o, "lumen_fraction", 0.7)),
                                  n=dict(getattr(o, "solute", {})))
                    )
            A0, _ = sphere_area_from_volume(cell.V); cell.A0 = A0
            cells.append(cell)

        bath = Bath(V=sum(c.V for c in cells),
                    n={"Na": float(getattr(sim, "external_concentration", 1500.0)),
                       "K": 0.0, "Cl": float(getattr(sim, "external_concentration", 1500.0)), "Imp": 0.0},
                    pressure=float(getattr(sim, "external_pressure", 1e4)),
                    temperature=float(getattr(sim, "temperature", 298.15)),
                    compressibility=float(getattr(sim, "bath_compressibility", 0.0)))
        return cls(cells, bath,
                   species=("Na","K","Cl","Imp"),
                   epsilon=float(getattr(sim, "epsilon", 1e-6)),
                   chars=[chr(97+i) for i in range(len(cells))],
                   width=int(getattr(sim, "bitbuffer", getattr(sim, "width", 80)).mask_size if hasattr(sim, "bitbuffer") else int(getattr(sim,"width",80))),
                   s_exprs=getattr(sim, "s_exprs", None),
                   p_exprs=getattr(sim, "p_exprs", None))


# Convenience adapters to mirror old salinepressure entry points
def run_saline_sim(sim, *, as_float: bool = False):
    """Legacy-compatible entry: attach a SalinePressureAPI onto sim and compute fractions.

    - Builds cellsim Cell/Bath objects from the legacy sim using SalinePressureAPI.from_legacy
    - Stores API instance on sim as sim.cs_api and engine as sim.engine_cs
    - Computes equilibrium fractions at t=0 and stores at sim.fractions
    - Returns proposals from sim.snap_cell_walls to keep behavior close to legacy
    """
    api = SalinePressureAPI.from_legacy(sim)
    sim.cs_api = api
    sim.engine_cs = api.engine

    # ensure leftmost/rightmost are defaulted for snapping
    for cell in tqdm(sim.cells, desc="cells", leave=False):
        if getattr(cell, "leftmost", None) is None:
            cell.leftmost = cell.left
        if getattr(cell, "rightmost", None) is None:
            cell.rightmost = cell.right - 1

    sim.fractions = api.equilibrium_fracs(0.0)

    # Legacy flow returned proposals after a snap to boundaries
    from src.transmogrifier.bitbitbuffer.helpers.cell_proposal import CellProposal
    proposals = [CellProposal(c) for c in tqdm(sim.cells, desc="cells", leave=False)]
    proposals = sim.snap_cell_walls(sim.cells, proposals)
    return proposals


def update_s_p_expressions(sim, cells, *, as_float: bool = False):
    from sympy import Float, Integer
    if as_float:
        sim.s_exprs = [Float(getattr(cell, "salinity", 0.0)) for cell in tqdm(cells, desc="cells", leave=False)]
        sim.p_exprs = [Float(getattr(cell, "pressure", 0.0)) for cell in tqdm(cells, desc="cells", leave=False)]
    else:
        sim.s_exprs = [Integer(getattr(cell, "salinity", 0)) for cell in tqdm(cells, desc="cells", leave=False)]
        sim.p_exprs = [Integer(getattr(cell, "pressure", 0)) for cell in tqdm(cells, desc="cells", leave=False)]


def equilibrium_fracs(sim, t: float):
    # ensure API exists
    if not hasattr(sim, "cs_api"):
        sim.cs_api = SalinePressureAPI.from_legacy(sim)
    return sim.cs_api.equilibrium_fracs(t)


def balance_system(sim, cells, bitbuffer, *args, **kwargs):
    # Lightweight legacy-compatible: ensure geometric fields and set concentrations/pressure.
    from math import pi
    from src.transmogrifier.bitbitbuffer.helpers.cell_proposal import CellProposal

    def sphere_area_from_volume_legacy(V: float):
        R = (3.0 * V / (4.0 * pi)) ** (1.0 / 3.0)
        return 4.0 * pi * R * R, R

    for c in tqdm(cells, desc="cells", leave=False):
        # establish basic geometric/mech fields if missing
        if not hasattr(c, "volume"):
            c.volume = float(c.right - c.left)
        if not hasattr(c, "initial_volume"):
            c.initial_volume = float(c.volume)
        if not hasattr(c, "A0"):
            c.A0, _ = sphere_area_from_volume_legacy(c.initial_volume)
        if not hasattr(c, "elastic_k"):
            c.elastic_k = 0.1
        if not hasattr(c, "visc_eta"):
            c.visc_eta = 0.0
        if not hasattr(c, "base_pressure"):
            c.base_pressure = 1e4#float(getattr(c, "pressure", 0.0))

        # concentration bookkeeping from legacy salinity
        sal = float(getattr(c, "salinity", 0.0))
        c.concentration = sal / max(c.volume, 1e-18)
        c.concentrations = {"Imp": c.concentration}

        # compute pressure from current strain (usually 0 at init)
        A_curr, R_curr = sphere_area_from_volume_legacy(c.volume)
        strain = max(A_curr / c.A0 - 1.0, 0.0)
        c.pressure = c.base_pressure + (2.0 * (c.elastic_k * strain) / max(R_curr, 1e-12))

    return [CellProposal(c) for c in tqdm(cells, desc="cells", leave=False)]


def run_balanced_saline_sim(sim, mode: str = "open", *, dt: float = 1, max_steps: int = 20,
                            tol_vol: float = 1e-9, tol_conc: float = 1e-9) -> list:
    """Step cellsim engine to equilibrium, sync derived state back, then return proposals.

    - Builds SalinePressureAPI from the legacy Simulator
    - Steps SalineEngine with adaptive dt until small relative volume changes and concentration differences
    - Copies pressure/volume and concentrations back to legacy cells
    - Computes fractions for the legacy integer-bar view and returns snap proposals
    """
    from src.transmogrifier.bitbitbuffer.helpers.cell_proposal import CellProposal

    api = SalinePressureAPI.from_legacy(sim)
    sim.cs_api = api
    sim.engine_cs = api.engine

    # 0) Preprocess: convert PID mask placements into organelles and a dynamic relocation mapping
    relocation_map = build_dynamic_pid_relocation(sim, api)
    # Communicate plan to caller; applying it is caller-controlled
    sim.relocation_map = relocation_map

    # Debug: pre-run snapshot (after organelle preprocessing), include relocation plan
    try:
        pre = api.snapshot_system(sim)
        if getattr(sim, "relocation_map", None):
            pre["relocation_map"] = {
                k: {
                    "sources": v.get("sources", []),
                    "destinations": v.get("destinations", []),
                }
                for k, v in sim.relocation_map.items()
            }
        logger.debug("saline pre-run system status:\n%s", json.dumps(pre, indent=2))
    except Exception as e:
        logger.debug("failed to build pre-run snapshot: %s", e)

    # Iterate to equilibrium
    dt_curr = float(dt)
    species = tuple(api.species)
    species_list = list(species)
    for step in tqdm(range(int(max_steps)), desc="steps", leave=False):
        vols_before = np.array([c.V for c in api.cells])
        dt_curr = api.engine.step(dt_curr)
        vols_after = np.array([c.V for c in api.cells])
        Cext = api.bath.conc(species_list)
        Cext_vec = np.array([Cext.get(sp, 0.0) for sp in species])

        # max relative volume change (vectorized)
        denom = np.where(np.abs(vols_after) > 1e-18, vols_after, 1e-18)
        max_rel = np.max(np.abs(vols_after - vols_before) / denom)

        # max concentration mismatch with bath (vectorized)
        V_free = np.maximum(vols_after, 1e-18)
        n_matrix = np.array([[c.n.get(sp, 0.0) for sp in species] for c in api.cells])
        Ci = n_matrix / V_free[:, None]
        max_dc = np.max(np.abs(Cext_vec - Ci))

        if max_rel < tol_vol and max_dc < tol_conc:
            break

    # Sync derived state back to legacy cells
    for legacy, cs in tqdm(zip(sim.cells, api.cells), desc="cells", total=len(sim.cells), leave=False):
        # volumes are useful for downstream bookkeeping
        legacy.volume = cs.V
        # pressure computed during engine step (from mechanics)
        legacy.pressure = getattr(cs, "pressure", getattr(legacy, "pressure", 0.0))
        # concentrations for convenience
        Vc = max(cs.V, 1e-18)
        conc = {sp: cs.n.get(sp, 0.0)/Vc for sp in species}
        legacy.concentrations = conc
        legacy.concentration = conc.get("Imp", conc.get("Na", 0.0))

    # Compute fractions at t=0 and produce proposals via snap
    sim.fractions = api.equilibrium_fracs(0.0)
    proposals = [CellProposal(c) for c in sim.cells]
    proposals = sim.snap_cell_walls(sim.cells, proposals)
    # Debug: post-run snapshot (include relocation plan for traceability)
    try:
        post = api.snapshot_system(sim)
        if getattr(sim, "relocation_map", None):
            post["relocation_map"] = {
                k: {
                    "sources": v.get("sources", []),
                    "destinations": v.get("destinations", []),
                }
                for k, v in sim.relocation_map.items()
            }
        logger.debug("saline post-run system status:\n%s", json.dumps(post, indent=2))
    except Exception as e:
        logger.debug("failed to build post-run snapshot: %s", e)
    # Return the relocation plan alongside proposals for upstream preservation if desired
    return proposals


def build_dynamic_pid_relocation(sim, api: SalinePressureAPI):
    """Create organelles from PID masks and propose dynamic index relocation per cell.

    Returns a dict keyed by cell.label with:
      { 'sources': [abs_indices...], 'destinations': [abs_indices...] }

    This mirrors current PID mask placements (set bits) to the compacted left side
    of each cell, preserving relative order. It's a placeholder policy that moves
    objects based on organelle occupancy without static gap insertions.
    """
    bitbuf = sim.bitbuffer
    label_to_cs = {getattr(leg, "label", f"c{i}"): cs for i, (leg, cs) in enumerate(zip(sim.cells, api.cells))}
    mapping: dict[str, dict[str, list[int]]] = {}

    for cell in tqdm(sim.cells, desc="cells", leave=False):
        pb = bitbuf.pid_buffers.get(cell.label)
        if pb is None:
            continue
        stride = max(1, getattr(cell, "stride", pb.domain_stride))
        left = cell.left
        right = cell.right
        slots = (right - left) // stride if stride > 0 else 0

        set_abs_indices: list[int] = []
        # Scan for set mask bits (occupied slots)
        for i in tqdm(range(slots), desc="slots", leave=False):
            abs_bit = left + i * stride
            idx = (abs_bit - pb.domain_left) // stride
            if 0 <= idx < pb.pids.mask_size and int(pb.pids[idx]) == 1:
                set_abs_indices.append(abs_bit)

        # Create organelles sized by occupancy fraction (still useful to engine)
        cs_cell = label_to_cs.get(cell.label)
        if cs_cell is not None and set_abs_indices:
            # approximate: one organelle spanning occupied count
            occ_slots = len(set_abs_indices)
            vol = float(occ_slots * stride)
            cs_cell.organelles.append(Organelle(volume_total=vol, lumen_fraction=0.0, n={"Imp": 0.0}, V_solid=vol, incompressible=True))

        # Build a simple left-pack relocation: move occupied slots to the left edge
        dst_indices: list[int] = []
        for k, _abs in enumerate(set_abs_indices):
            dst_indices.append(left + k * stride)

        if set_abs_indices:
            mapping[cell.label] = {
                "sources": set_abs_indices,
                "destinations": dst_indices,
            }

    return mapping
