from typing import Iterable, List, Optional
import math
import numpy as np
from ..core.geometry import sphere_area_from_volume
from ..core.numerics import clamp_nonneg, adapt_dt
from ..core.units import R as RGAS
from ..mechanics.tension import laplace_pressure
from ..transport.kedem_katchalsky import arrhenius
from ..organelles.inner_loop import inner_exchange, cytosol_free_volume
from ..core import checks
from ..data.state import Cell, Bath
from ..mechanics.provider import MechanicsProvider, MechanicsSnapshot
from ..transport.pumps import (
    na_k_atpase_constant,
    na_k_atpase_saturating,
)

class SalineEngine:
    def __init__(
        self,
        cells: List[Cell],
        bath: Bath,
        species: Iterable[str] = ("Na", "K", "Cl", "Imp"),
        *,
        enable_energy_check: bool = False,
        enable_checks: bool = False,
        mechanics_provider: Optional[MechanicsProvider] = None,
        ):
        self.cells = cells
        self.bath = bath
        self.species = tuple(species)
        self.enable_energy_check = enable_energy_check
        self.enable_checks = enable_checks
        self.mechanics_provider = mechanics_provider

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
        species_list = list(self.species)
        n_cells = len(self.cells)
        n_species = len(species_list)

        # Totals before step using array reductions
        cell_n = np.array([[c.n.get(sp, 0.0) for sp in species_list] for c in self.cells], dtype=float)
        organelle_n = np.array(
            [[sum(o.n.get(sp, 0.0) for o in getattr(c, "organelles", [])) for sp in species_list] for c in self.cells],
            dtype=float,
        )
        bath_n = np.array([self.bath.n.get(sp, 0.0) for sp in species_list], dtype=float)
        totals_before_arr = bath_n + cell_n.sum(axis=0) + organelle_n.sum(axis=0)
        totals_before = {sp: totals_before_arr[i] for i, sp in enumerate(species_list)}

        # Inner organelle exchange per cell
        for c in self.cells:
            inner_exchange(c, T, dt, self.species, Rgas=RGAS)

        # Gather state arrays post inner exchange
        V = np.array([c.V for c in self.cells], dtype=float)
        n = np.array([[c.n.get(sp, 0.0) for sp in species_list] for c in self.cells], dtype=float)
        A0 = np.array([c.A0 for c in self.cells], dtype=float)
        elastic_k = np.array([c.elastic_k for c in self.cells], dtype=float)
        visc_eta = np.array([c.visc_eta for c in self.cells], dtype=float)
        eps_prev = np.array([c._prev_eps for c in self.cells], dtype=float)
        base_pressure = np.array([c.base_pressure for c in self.cells], dtype=float)
        Lp0 = np.array([c.Lp0 for c in self.cells], dtype=float)
        Ea_Lp = np.array([c.Ea_Lp if c.Ea_Lp is not None else np.nan for c in self.cells], dtype=float)
        Ps0 = np.array([[c.Ps0.get(sp, 0.0) for sp in species_list] for c in self.cells], dtype=float)
        Ea_Ps = np.array([[c.Ea_Ps.get(sp) if c.Ea_Ps.get(sp) is not None else np.nan for sp in species_list] for c in self.cells], dtype=float)
        sigma = np.array([[c.sigma.get(sp, 1.0) for sp in species_list] for c in self.cells], dtype=float)

        # Mechanics and anchoring
        dP_tension, eps = laplace_pressure(A0, V, elastic_k, visc_eta, eps_prev, dt)
        dP_anchor = np.array(
            [
                np.sum([
                    o.anchor_stiffness * (eps[i] - o.eps_ref)
                    for o in getattr(self.cells[i], "organelles", [])
                    if o.anchor_stiffness > 0.0 and math.isfinite(o.anchor_stiffness)
                ])
                for i in range(n_cells)
            ],
            dtype=float,
        )
        P_i = base_pressure + dP_tension + dP_anchor
        for i in range(n_cells):
            self.cells[i]._prev_eps = float(eps[i])

        # Optional mechanics provider overrides
        mech: Optional[MechanicsSnapshot] = None
        if self.mechanics_provider is not None:
            try:
                self.mechanics_provider.sync(self.cells, self.bath)
                mech = self.mechanics_provider.step(dt)
            except Exception:
                mech = None
        areas_override = np.asarray(mech.get("areas", []), dtype=float) if isinstance(mech, dict) else np.empty(0)
        pressures_override = np.asarray(mech.get("pressures", []), dtype=float) if isinstance(mech, dict) else np.empty(0)
        if pressures_override.size:
            P_i[: min(n_cells, pressures_override.size)] = pressures_override[: n_cells]

        # Concentrations
        V_free = np.array([cytosol_free_volume(c) for c in self.cells], dtype=float)
        Cint = n / V_free[:, None]
        Cext_vec = np.array([self.bath.conc(species_list)[sp] for sp in species_list], dtype=float)

        # Geometry
        A, _ = sphere_area_from_volume(V)
        if areas_override.size:
            A[: min(n_cells, areas_override.size)] = areas_override[: n_cells]

        # Permeabilities
        Lp = np.array(
            [arrhenius(Lp0[i], None if np.isnan(Ea_Lp[i]) else Ea_Lp[i], T) for i in range(n_cells)],
            dtype=float,
        )
        Ps = np.array(
            [
                [arrhenius(Ps0[i, j], None if np.isnan(Ea_Ps[i, j]) else Ea_Ps[i, j], T) for j in range(n_species)]
                for i in range(n_cells)
            ],
            dtype=float,
        )

        # Fluxes (vectorised over cells)
        osm = np.sum(sigma * RGAS * T * (Cint - Cext_vec[None, :]), axis=1)
        Jv = Lp * A * ((P_i - self.bath.pressure) - osm)
        dV = -Jv
        Js = Ps * A[:, None] * (Cint - Cext_vec[None, :]) + (1.0 - sigma) * Cint * Jv[:, None]
        dS = -Js

        # Na/K pump updates (batched)
        J_pump = np.zeros(n_cells, dtype=float)
        idx_Na = species_list.index("Na") if "Na" in species_list else None
        idx_K = species_list.index("K") if "K" in species_list else None
        for i, c in enumerate(self.cells):
            if getattr(c, "J_pump", 0.0) > 0.0:
                J_pump[i] = na_k_atpase_constant(c.J_pump)
            elif getattr(c, "pump_enabled", False):
                C_Nai = Cint[i, idx_Na] if idx_Na is not None else 0.0
                C_Ko = Cext_vec[idx_K] if idx_K is not None else 0.0
                J_pump[i] = na_k_atpase_saturating(
                    C_Nai=C_Nai,
                    C_Ko=C_Ko,
                    A=A[i],
                    Jmax=getattr(c, "pump_Jmax", 0.0),
                    Km_Nai=getattr(c, "pump_Km_Nai", 10.0),
                    Km_Ko=getattr(c, "pump_Km_Ko", 1.5),
                    eps=eps[i],
                    alpha_tension=getattr(c, "pump_alpha_tension", 0.0),
                )
        if idx_Na is not None:
            dS[:, idx_Na] -= 3.0 * J_pump * dt
        if idx_K is not None:
            dS[:, idx_K] += 2.0 * J_pump * dt

        # Optional energy check
        if self.enable_energy_check:
            for i, c in enumerate(self.cells):
                if J_pump[i] <= 0.0:
                    dS_cell = {sp: dS[i, j] for j, sp in enumerate(species_list)}
                    Cint_dict = {sp: Cint[i, j] for j, sp in enumerate(species_list)}
                    Cext_dict = {sp: Cext_vec[j] for j, sp in enumerate(species_list)}
                    checks.assert_passive_no_energy(c, self.bath, dS_cell, Cint_dict, Cext_dict, self.species, T)

        # Apply updates with masks
        V_occ = np.array(
            [sum(getattr(o, "V_solid", 0.0) + o.V_lumen() for o in c.organelles) for c in self.cells],
            dtype=float,
        )
        V_min = np.maximum(V_occ, 1e-18)
        V_next = V + dV
        below = V_next < V_min
        V_next[below] = V_min[below]
        dV = V_next - V
        V = np.maximum(V_next, 0.0)

        n_new = n + dS
        neg_mask = n_new < 0.0
        dS[neg_mask] = -n[neg_mask]
        n_new = np.maximum(n_new, 0.0)
        n = n_new

        bath_n -= dS.sum(axis=0)
        bath_n = np.maximum(bath_n, 0.0)
        if getattr(self.bath, "compressibility", 0.0) > 0.0:
            self.bath.V = clamp_nonneg(self.bath.V - dV.sum())

        max_rel = float(np.max(np.abs(dV) / np.maximum(V, 1e-18)))
        sum_dV = float(np.sum(dV))

        # Assign arrays back to objects
        for i, c in enumerate(self.cells):
            c.V = float(V[i])
            for j, sp in enumerate(species_list):
                c.n[sp] = float(n[i, j])
            c.pressure = float(P_i[i])
            c.concentrations = {sp: c.n[sp] / max(c.V, 1e-18) for sp in species_list}
            c.concentration = c.concentrations.get("Imp", 0.0)
            occ_post = sum(getattr(o, "V_solid", 0.0) + o.V_lumen() for o in c.organelles)
            assert c.V + 1e-18 >= occ_post

        for j, sp in enumerate(species_list):
            self.bath.n[sp] = float(bath_n[j])

        if getattr(self.bath, "compressibility", 0.0) > 0.0:
            self.bath.pressure += -(sum_dV / (self.bath.compressibility * max(self.bath.V, 1e-18)))

        if self.enable_checks:
            checks.assert_nonneg(self.cells, self.bath, self.species)
            checks.assert_mass_conserved(self.cells, self.bath, self.species, totals_before)

        return adapt_dt(dt, max_rel)
