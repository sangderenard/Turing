from typing import Iterable, List, Optional
import math
import numpy as np
from ..core.geometry import sphere_area_from_volume
from ..core.numerics import adapt_dt
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
        self.species_index = {sp: i for i, sp in enumerate(self.species)}
        self.enable_energy_check = enable_energy_check
        self.enable_checks = enable_checks
        self.mechanics_provider = mechanics_provider

        # Initialize immutable/default arrays from input cells without coercing to Python floats
        n_cells = len(self.cells)
        species_list = list(self.species)

        # Make sure geometric/mech baseline exists
        for c in self.cells:
            c.set_initial_A0_if_missing()

        # Core state as numpy arrays (authoritative during the simulation)
        self.V = np.asarray([getattr(c, "V", 0.0) for c in self.cells], dtype=float)
        self.A0 = np.asarray([getattr(c, "A0", 0.0) for c in self.cells], dtype=float)
        self.elastic_k = np.asarray([getattr(c, "elastic_k", 0.0) for c in self.cells], dtype=float)
        self.visc_eta = np.asarray([getattr(c, "visc_eta", 0.0) for c in self.cells], dtype=float)
        self._prev_eps = np.zeros(n_cells, dtype=float)
        self.base_pressure = np.asarray([getattr(c, "base_pressure", 0.0) for c in self.cells], dtype=float)
        self.Lp0 = np.asarray([getattr(c, "Lp0", 0.0) for c in self.cells], dtype=float)
        self.Ea_Lp = np.asarray([getattr(c, "Ea_Lp", np.nan) if getattr(c, "Ea_Lp", None) is not None else np.nan for c in self.cells], dtype=float)
        self.Ps0 = np.asarray([[getattr(c, "Ps0", {}).get(sp, 0.0) for sp in species_list] for c in self.cells], dtype=float)
        self.Ea_Ps = np.asarray([[getattr(c, "Ea_Ps", {}).get(sp) if getattr(c, "Ea_Ps", {}).get(sp) is not None else np.nan for sp in species_list] for c in self.cells], dtype=float)
        self.sigma = np.asarray([[getattr(c, "sigma", {}).get(sp, 1.0) for sp in species_list] for c in self.cells], dtype=float)
        # moles per species (cells)
        self.n = np.asarray([[getattr(c, "n", {}).get(sp, 0.0) for sp in species_list] for c in self.cells], dtype=float)
        # pump controls
        self.pump_enabled = np.asarray([getattr(c, "pump_enabled", False) for c in self.cells], dtype=bool)
        self.J_pump_const = np.asarray([getattr(c, "J_pump", 0.0) for c in self.cells], dtype=float)
        self.pump_Jmax = np.asarray([getattr(c, "pump_Jmax", 0.0) for c in self.cells], dtype=float)
        self.pump_Km_Nai = np.asarray([getattr(c, "pump_Km_Nai", 10.0) for c in self.cells], dtype=float)
        self.pump_Km_Ko = np.asarray([getattr(c, "pump_Km_Ko", 1.5) for c in self.cells], dtype=float)
        self.pump_alpha_tension = np.asarray([getattr(c, "pump_alpha_tension", 0.0) for c in self.cells], dtype=float)

        # Bath state as arrays
        self.bath_n = np.asarray([getattr(self.bath, "n", {}).get(sp, 0.0) for sp in species_list], dtype=float)
        self.bath_pressure = getattr(self.bath, "pressure", 0.0)
        self.bath_temperature = getattr(self.bath, "temperature", 298.15)
        self.bath_compressibility = getattr(self.bath, "compressibility", 0.0)
        self.bath_V = getattr(self.bath, "V", 1.0)

        # Expose last computed arrays for external readers
        self.P_i = np.zeros(n_cells, dtype=float)
        self.osmotic_pressure = np.zeros(n_cells, dtype=float)
        self.A = np.zeros(n_cells, dtype=float)

        # If a mechanics provider is attached, ensure it's initialized with
        # the current object graph so any internal world (e.g., softbody
        # hierarchy) is constructed even when the provider also supports an
        # array-fast sync path. Without this, providers that implement only
        # a no-op sync_arrays when uninitialized would never build their
        # world, leading downstream demos/viewers to see no data.
        if self.mechanics_provider is not None and hasattr(self.mechanics_provider, "sync"):
            try:
                self.mechanics_provider.sync(self.cells, self.bath)  # type: ignore[attr-defined]
            except Exception:
                print(f"Warning: Mechanics provider {self.mechanics_provider} failed to sync on initialization. ")
                raise

    def copy_shallow(self):
        return {
            "V": self.V.copy(),
            "n": self.n.copy(),
            "bath_n": self.bath_n.copy(),
            "bath_pressure": self.bath_pressure,
            "bath_temperature": self.bath_temperature,
            "bath_V": self.bath_V,
            "_prev_eps": self._prev_eps.copy(),
        }

    def restore(self, saved) -> None:
        self.V[:] = saved["V"]
        self.n[:] = saved["n"]
        self.bath_n[:] = saved["bath_n"]
        self.bath_pressure = saved["bath_pressure"]
        self.bath_temperature = saved["bath_temperature"]
        self.bath_V = saved["bath_V"]
        self._prev_eps[:] = saved["_prev_eps"]

    def step(self, dt: float, *, use_adapt: bool = True, hooks=None) -> float:
        T = self.bath_temperature
        species_list = list(self.species)
        n_cells = self.V.shape[0]
        n_species = len(species_list)

        # Totals before step (arrays only)
        # Note: organelle tracking is disabled in array mode to keep the path numpy-only.
        totals_before_arr = self.bath_n + self.n.sum(axis=0)
        totals_before = {sp: totals_before_arr[i] for i, sp in enumerate(species_list)}

        # Mechanics and anchoring
        dP_tension, eps = laplace_pressure(self.A0, self.V, self.elastic_k, self.visc_eta, self._prev_eps, dt)
        # anchoring via organelles omitted in array-first path (requires object graph)
        dP_anchor = np.zeros(n_cells, dtype=float)
        P_i = self.base_pressure + dP_tension + dP_anchor
        self._prev_eps = eps

        # Optional mechanics provider overrides via array sync
        mech: Optional[MechanicsSnapshot] = None
        if self.mechanics_provider is not None:
            # Prefer array-based sync to avoid object conversions
            if hasattr(self.mechanics_provider, "sync_arrays"):
                idx_imp = self.species_index.get("Imp", None)
                imp = self.n[:, idx_imp] if idx_imp is not None else np.zeros(n_cells)
                try:
                    self.mechanics_provider.sync_arrays(
                        V=self.V,
                        elastic_k=self.elastic_k,
                        imp=imp,
                        bath_pressure=self.bath_pressure,
                        bath_temperature=self.bath_temperature,
                    )
                except Exception:
                    pass
            else:
                try:
                    # Fallback (may coerce types inside provider)
                    self.mechanics_provider.sync(self.cells, self.bath)
                except Exception:
                    pass
            try:
                mech = self.mechanics_provider.step(dt, hooks=hooks)
            except Exception:
                mech = None

        areas_override = np.asarray(mech.get("areas", []), dtype=float) if isinstance(mech, dict) else np.empty(0)
        pressures_override = np.asarray(mech.get("pressures", []), dtype=float) if isinstance(mech, dict) else np.empty(0)
        if pressures_override.size:
            P_i[: min(n_cells, pressures_override.size)] = pressures_override[: n_cells]

        # Concentrations (use full volume by default; organelle excluded-volume path requires object graph)
        V_free = np.maximum(self.V, 1e-18)
        Cint = self.n / V_free[:, None]
        # Bath concentrations vector
        Cext_vec = self.bath_n / max(self.bath_V, 1e-18)

        # Geometry
        A, _ = sphere_area_from_volume(self.V)
        if areas_override.size:
            A[: min(n_cells, areas_override.size)] = areas_override[: n_cells]

        # Permeabilities
        Lp = np.asarray([arrhenius(self.Lp0[i], None if np.isnan(self.Ea_Lp[i]) else self.Ea_Lp[i], T) for i in range(n_cells)], dtype=float)
        Ps = np.asarray(
            [
                [arrhenius(self.Ps0[i, j], None if np.isnan(self.Ea_Ps[i, j]) else self.Ea_Ps[i, j], T) for j in range(n_species)]
                for i in range(n_cells)
            ],
            dtype=float,
        )

        # Fluxes (vectorised over cells)
        osm = np.sum(self.sigma * RGAS * T * (Cint - Cext_vec[None, :]), axis=1)
        Jv = Lp * A * ((P_i - self.bath_pressure) - osm)
        dV = -Jv
        Js = Ps * A[:, None] * (Cint - Cext_vec[None, :]) + (1.0 - self.sigma) * Cint * Jv[:, None]
        dS = -Js

        # Na/K pump updates (batched, vectorised where possible)
        J_pump = np.zeros(n_cells, dtype=float)
        idx_Na = self.species_index.get("Na")
        idx_K = self.species_index.get("K")
        if idx_Na is not None or idx_K is not None:
            # Refresh constant pump rates from cells each step
            self.J_pump_const = np.asarray(
                [getattr(c, "J_pump", 0.0) for c in self.cells], dtype=float
            )
            # Constant-rate for cells with J_pump > 0
            mask_const = self.J_pump_const > 0.0
            if np.any(mask_const):
                J_pump[mask_const] = na_k_atpase_constant(self.J_pump_const[mask_const])
            # Saturating for enabled pumps
            mask_sat = (~mask_const) & self.pump_enabled
            if np.any(mask_sat):
                C_Nai = Cint[mask_sat, idx_Na] if idx_Na is not None else np.zeros(np.count_nonzero(mask_sat))
                C_Ko = Cext_vec[idx_K] if idx_K is not None else 0.0
                # Vectorised saturating update across enabled subset
                A_sub = A[mask_sat]
                J_pump[mask_sat] = na_k_atpase_saturating(
                    C_Nai=C_Nai,
                    C_Ko=C_Ko,
                    A=A_sub,
                    Jmax=self.pump_Jmax[mask_sat],
                    Km_Nai=self.pump_Km_Nai[mask_sat],
                    Km_Ko=self.pump_Km_Ko[mask_sat],
                    eps=eps[mask_sat],
                    alpha_tension=self.pump_alpha_tension[mask_sat],
                )
        if idx_Na is not None:
            dS[:, idx_Na] -= 3.0 * J_pump * dt
        if idx_K is not None:
            dS[:, idx_K] += 2.0 * J_pump * dt

        # Optional energy check (requires object graph; keep minimal dict construction)
        if self.enable_energy_check:
            for i in range(n_cells):
                if J_pump[i] <= 0.0:
                    dS_cell = {sp: dS[i, j] for j, sp in enumerate(species_list)}
                    Cint_dict = {sp: Cint[i, j] for j, sp in enumerate(species_list)}
                    Cext_dict = {sp: Cext_vec[j] for j, sp in enumerate(species_list)}
                    checks.assert_passive_no_energy(self.cells[i], self.bath, dS_cell, Cint_dict, Cext_dict, self.species, T)

        # Let bath enforce thermodynamic constraints and update its state
        dV, dS = self.bath.apply_physics(dV, dS, {"species": species_list})
        self.bath_pressure = float(self.bath.pressure)
        self.bath_temperature = float(self.bath.temperature)
        self.bath_V = float(self.bath.V)
        self.bath_n = np.asarray([self.bath.n.get(sp, 0.0) for sp in species_list], dtype=float)

        # Apply updates with masks (array-only)
        V_min = np.full(n_cells, 1e-18, dtype=float)
        V_next = self.V + dV
        below = V_next < V_min
        V_next = np.where(below, V_min, V_next)
        dV = V_next - self.V
        self.V = np.maximum(V_next, 0.0)

        n_new = self.n + dS
        neg_mask = n_new < 0.0
        dS[neg_mask] = -self.n[neg_mask]
        self.n = np.maximum(n_new, 0.0)

        max_rel = float(np.max(np.abs(dV) / np.maximum(self.V, 1e-18)))

        # Update cached observables
        self.P_i = P_i
        self.osmotic_pressure = osm
        self.A = A

        # Scatter array state back to object graph for external observers
        for i, c in enumerate(self.cells):
            c.V = float(self.V[i])
            for j, sp in enumerate(species_list):
                c.n[sp] = float(self.n[i, j])
        for i, sp in enumerate(species_list):
            self.bath.n[sp] = float(self.bath_n[i])
        self.bath.pressure = float(self.bath_pressure)
        self.bath.temperature = float(self.bath_temperature)
        self.bath.V = float(self.bath_V)

        if self.enable_checks:
            # These checks still reference object graph for messages; can be disabled if desired
            checks.assert_nonneg(self.cells, self.bath, self.species)
            checks.assert_mass_conserved(self.cells, self.bath, self.species, totals_before)

        if use_adapt:
            return adapt_dt(dt, max_rel)
        return dt
