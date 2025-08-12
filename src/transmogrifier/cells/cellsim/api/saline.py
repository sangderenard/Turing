# cellsim/api/saline.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional
from sympy import lambdify, symbols, Integer, Float
from ..engine.saline import SalineEngine
from ..data.state import Cell, Bath, Organelle
from ..core.geometry import sphere_area_from_volume

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
        for c in self.cells:
            if c.A0 <= 0.0:
                A0, _ = sphere_area_from_volume(c.V); c.A0 = A0
        self.engine = SalineEngine(self.cells, self.bath, species=self.species)

    # ---- legacy “view”: equilibrium fractions & bar ----
    def equilibrium_fracs(self, t: float) -> List[float]:
        s_vals = [f(t) for f in self.s_funcs]
        p_vals = [f(t) for f in self.p_funcs]
        r = []
        eps = self.epsilon
        for si, pi in zip(s_vals, p_vals):
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
        for i, q in enumerate(quotas):
            if q <= 1.0 and self.int_alloc.protect_under_one:
                cost = float('inf')
            else:
                cost = 1 - (q - floor(q))
            costs.append((cost, i))
        for _, idx in sorted(costs, key=lambda x: x[0])[:K]:
            ceilings[idx] -= 1
        return ceilings

    def _integer_allocate_adams(self, fracs, W):
        from math import floor, ceil
        quotas = [f*W for f in fracs]
        ceilings = [max(1, ceil(q)) if self.int_alloc.bump_under_one else ceil(q) for q in quotas]
        K = sum(ceilings) - W
        if K <= 0: return ceilings
        costs = []
        for i, q in enumerate(quotas):
            if q < 1.0 and self.int_alloc.protect_under_one:
                cost = float('inf')
            else:
                cost = 1 - (q - floor(q))
            costs.append((cost, i))
        for _, idx in sorted(costs, key=lambda x: x[0])[:K]:
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
        for legacy in sim.cells:
            V = float(legacy.right - legacy.left)
            cell = Cell(V=V,
                        n={"Imp": float(getattr(legacy, "salinity", 0.0)),
                           "Na": 0.0, "K": 0.0, "Cl": 0.0},
                        base_pressure=float(getattr(legacy, "pressure", 0.0)),
                        elastic_k=float(getattr(legacy, "elastic_k", 0.1)),
                        visc_eta=float(getattr(legacy, "visc_eta", 0.0)))
            # organelles if present
            if hasattr(legacy, "organelles"):
                for o in legacy.organelles:
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
    for cell in sim.cells:
        if getattr(cell, "leftmost", None) is None:
            cell.leftmost = cell.left
        if getattr(cell, "rightmost", None) is None:
            cell.rightmost = cell.right - 1

    sim.fractions = api.equilibrium_fracs(0.0)

    # Legacy flow returned proposals after a snap to boundaries
    from src.transmogrifier.bitbitbuffer.helpers.cell_proposal import CellProposal
    proposals = [CellProposal(c) for c in sim.cells]
    proposals = sim.snap_cell_walls(sim.cells, proposals)
    return proposals


def update_s_p_expressions(sim, cells, *, as_float: bool = False):
    from sympy import Float, Integer
    if as_float:
        sim.s_exprs = [Float(getattr(cell, "salinity", 0.0)) for cell in cells]
        sim.p_exprs = [Float(getattr(cell, "pressure", 0.0)) for cell in cells]
    else:
        sim.s_exprs = [Integer(getattr(cell, "salinity", 0)) for cell in cells]
        sim.p_exprs = [Integer(getattr(cell, "pressure", 0)) for cell in cells]


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

    for c in cells:
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
            c.base_pressure = float(getattr(c, "pressure", 0.0))

        # concentration bookkeeping from legacy salinity
        sal = float(getattr(c, "salinity", 0.0))
        c.concentration = sal / max(c.volume, 1e-18)
        c.concentrations = {"Imp": c.concentration}

        # compute pressure from current strain (usually 0 at init)
        A_curr, R_curr = sphere_area_from_volume_legacy(c.volume)
        strain = max(A_curr / c.A0 - 1.0, 0.0)
        c.pressure = c.base_pressure + (2.0 * (c.elastic_k * strain) / max(R_curr, 1e-12))

    return [CellProposal(c) for c in cells]


def run_balanced_saline_sim(sim, mode: str = "open", *, dt: float = 1e-3, max_steps: int = 200000,
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

    # 0) Preprocess: convert PID mask placements into organelles and relocation plan
    relocation_plan = preprocess_pid_masks_to_organelles(sim, api)
    # Communicate plan to caller; applying it is caller-controlled
    sim.relocation_plan = relocation_plan

    # Iterate to equilibrium
    dt_curr = float(dt)
    species = api.species
    for step in range(int(max_steps)):
        vols_before = [c.V for c in api.cells]
        dt_curr = api.engine.step(dt_curr)
        vols_after = [c.V for c in api.cells]
        # max relative volume change
        max_rel = 0.0
        for vb, va in zip(vols_before, vols_after):
            denom = va if abs(va) > 1e-18 else 1e-18
            rel = abs(va - vb) / denom
            if rel > max_rel:
                max_rel = rel
        # max concentration mismatch with bath
        Cext = api.bath.conc(list(species))
        max_dc = 0.0
        for c in api.cells:
            V_free = max(c.V, 1e-18)
            for sp in species:
                Ci = c.n.get(sp, 0.0) / V_free
                Ce = Cext.get(sp, 0.0)
                diff = abs(Ce - Ci)
                if diff > max_dc:
                    max_dc = diff
        if max_rel < tol_vol and max_dc < tol_conc:
            break

    # Sync derived state back to legacy cells
    for legacy, cs in zip(sim.cells, api.cells):
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
    # Return the relocation plan alongside proposals for upstream preservation if desired
    return proposals


def preprocess_pid_masks_to_organelles(sim, api: SalinePressureAPI):
    """Create organelles from PID masks and produce a stride-aligned relocation plan.

    - For each cell, read PIDBuffer mask bits as stride slots
    - Group contiguous set bits into organelle candidates (each 1 stride wide per slot)
    - Create Organelle entries on the corresponding cellsim Cell with volume_total proportional to stride count
    - Build a relocation plan list of tuples (label, offset, size_bits) where size_bits can be negative to shrink
    - The calling code is responsible for applying the plan using build_metadata/expand.
    """
    bitbuf = sim.bitbuffer
    stride_map = {c.label: c.stride for c in sim.cells}
    # Ensure each legacy cell maps to its cellsim partner by index
    label_to_cs = {getattr(leg, "label", f"c{i}"): cs for i, (leg, cs) in enumerate(zip(sim.cells, api.cells))}
    relocation_events = []

    for cell in sim.cells:
        pb = bitbuf.pid_buffers.get(cell.label)
        if pb is None:
            continue
        stride = max(1, getattr(cell, "stride", pb.domain_stride))
        left = cell.left
        width = (cell.right - cell.left)
        slots = width // stride if stride > 0 else 0
        # Scan pid mask plane for set bits within this domain
        set_slots = []
        for i in range(slots):
            abs_bit = left + i * stride
            # Map abs_bit to pid-index space
            idx = (abs_bit - pb.domain_left) // stride
            if 0 <= idx < pb.pids.mask_size and int(pb.pids[idx]) == 1:
                set_slots.append(i)

        # Group contiguous slots
        groups = []
        start = None
        prev = None
        for s in set_slots:
            if start is None:
                start = prev = s
            elif s == prev + 1:
                prev = s
            else:
                groups.append((start, prev))
                start = prev = s
        if start is not None:
            groups.append((start, prev))

        # Create organelles and craft relocation intents
        cs_cell = label_to_cs.get(cell.label)
        for g0, g1 in groups:
            slot_count = g1 - g0 + 1
            vol = float(slot_count * stride)
            if cs_cell is not None:
                cs_cell.organelles.append(Organelle(volume_total=vol, lumen_fraction=0.0, n={"Imp": 0.0}))
            # Intent: reserve space for organelle movement; here we record as contraction (negative) at the left edge of group,
            # to be respected by upstream planner if it chooses to relocate payload before expansion
            abs_off = left + g0 * stride
            relocation_events.append((cell.label, abs_off, -slot_count * stride))

    return relocation_events
