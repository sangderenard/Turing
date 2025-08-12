from typing import Iterable, List
import math
from ..core.geometry import sphere_area_from_volume
from ..core.numerics import clamp_nonneg, adapt_dt
from ..core.units import R as RGAS
from ..mechanics.tension import laplace_pressure
from ..transport.kedem_katchalsky import arrhenius, fluxes
from ..transport.pumps import na_k_pump_flux
from ..organelles.inner_loop import inner_exchange, cytosol_free_volume
from ..core import checks
from ..data.state import Cell, Bath

class SalineEngine:
    def __init__(self, cells: List[Cell], bath: Bath, species: Iterable[str] = ("Na","K","Cl","Imp"), *, pump_rate: float = 0.0, enable_energy_check: bool = False, enable_checks: bool = False):
        self.cells = cells
        self.bath = bath
        self.species = tuple(species)
        self.pump_rate = float(pump_rate)
        self.enable_energy_check = enable_energy_check
        self.enable_checks = enable_checks

        for c in self.cells:
            c.set_initial_A0_if_missing()
            # ensure species keys exist
            for sp in self.species:
                c.n.setdefault(sp, 0.0)

            for o in getattr(c, "organelles", []):
                for sp in self.species:
                    o.n.setdefault(sp, 0.0)

        for sp in self.species:
            self.bath.n.setdefault(sp, 0.0)


    def step(self, dt: float) -> float:
        T = self.bath.temperature
        max_rel = 0.0
        sum_dV = 0.0
        totals_before = {}
        for sp in self.species:
            total = self.bath.n.get(sp, 0.0)
            for c in self.cells:
                total += c.n.get(sp, 0.0)
                for o in getattr(c, "organelles", []):
                    total += o.n.get(sp, 0.0)
            totals_before[sp] = total

        # Precompute bath concentrations
        Cext = self.bath.conc(list(self.species))

        for c in self.cells:
            # 1) inner organelle exchange (does not touch c.V)
            inner_exchange(c, T, dt, self.species, Rgas=RGAS)

            # 2) mechanics + anchoring
            dP_tension, eps = laplace_pressure(c.A0, c.V, c.elastic_k, c.visc_eta, c._prev_eps, dt)
            dP_anchor = 0.0
            for o in c.organelles:
                if o.anchor_stiffness > 0.0 and math.isfinite(o.anchor_stiffness):
                    dP_anchor += o.anchor_stiffness * (eps - o.eps_ref)
            c._prev_eps = eps
            P_i = c.base_pressure + dP_tension + dP_anchor

            # 3) outer flux (cell ↔ bath): use cytosol free volume for Cint
            V_free = cytosol_free_volume(c)
            Cint = {sp: c.n.get(sp,0.0)/V_free for sp in self.species}

            A, R_c = sphere_area_from_volume(c.V)
            Lp = arrhenius(c.Lp0, c.Ea_Lp, T)

            dV_cell, dS_cell = fluxes(
                comp_left=c, comp_right=self.bath, species=self.species,
                Lp=Lp, Ps=c.Ps0, sigma=c.sigma, A=A, T=T, Rgas=RGAS,
                C_left_override=Cint, C_right_override=Cext,
                Jv_pressure_term=(self.bath.pressure - P_i)
            )

            # pump contribution (affects species only)
            if self.pump_rate > 0.0:
                pf = na_k_pump_flux(c, self.bath, A=A, rate=self.pump_rate, dt=dt)
                for sp, dS in pf.items():
                    dS_cell[sp] = dS_cell.get(sp, 0.0) + dS

            if self.enable_energy_check and self.pump_rate <= 0.0:
                checks.assert_passive_no_energy(c, self.bath, dS_cell, Cint, Cext, self.species, T)

            # Apply
            c.V = clamp_nonneg(c.V + dV_cell)
            for sp, dS in dS_cell.items():
                curr = c.n.get(sp, 0.0)
                new = curr + dS
                if new < 0.0:
                    dS = -curr
                    new = 0.0
                c.n[sp] = clamp_nonneg(new)
                self.bath.n[sp] = clamp_nonneg(self.bath.n.get(sp,0.0) - dS)
            self.bath.V = clamp_nonneg(self.bath.V - dV_cell)

            rel = abs(dV_cell) / max(c.V, 1e-18)
            if rel > max_rel: max_rel = rel
            sum_dV += dV_cell

            # record a convenient derived pressure (for UI)
            c.pressure = P_i

        # optional bath pressure update via compressibility
        if getattr(self.bath, "compressibility", 0.0) > 0.0:
            self.bath.pressure += -self.bath.compressibility * (sum_dV / max(self.bath.V, 1e-18))

        # invariant checks
        if self.enable_checks:
            checks.assert_nonneg(self.cells, self.bath, self.species)
            checks.assert_mass_conserved(self.cells, self.bath, self.species, totals_before)

        # adapt dt suggestion back to caller
        return adapt_dt(dt, max_rel)
