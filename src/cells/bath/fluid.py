"""Fluid mechanics and thermodynamics engine for cell simulations.

This module defines :class:`Bath`, a general-purpose bulk fluid model used to
represent any aqueous domain inside or outside cells.  Despite the colloquial
name "bath", the object is not limited to extracellular media; it can model
cytoplasm, extracellular space, or stand-alone fluid simulations.

``Bath`` accepts proposed volume and solute fluxes from other engines (e.g., the
osmotic engine or soft-body mechanics) and enforces global physical constraints
such as throughput limits, compressibility, and thermal response.  It updates
its internal state and returns corrected fluxes without duplicating the work of
those engines; instead it acts as a minimal fluid mechanics/thermodynamics core
that keeps the simulation grounded in physical reality.

This module is intentionally lightweight yet extensible, providing a foundation
for future fluid models across the project.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Bath:
    r"""Bulk fluid environment for simulated cells.

    Attributes mirror a minimal compartment with bulk thermodynamic parameters.
    The :meth:`conc` method mirrors the :class:`Compartment` interface used
    elsewhere in cellsim.

    Notes
    -----
    Water is treated with mild compressibility :math:`\Delta V = \kappa V\Delta P`
    (:math:`\kappa` typically :math:`4.5\times10^{-10}\,\mathrm{Pa^{-1}}`).
    Thermal response follows ``Q = \rho c_p V \Delta T`` with density modelled as
    ``\rho(T) = \rho_0 [1 - \beta (T - T_0)]`` where ``\beta\approx2.07\times10^{-4}``
    ``\text{K}^{-1}`` and ``T_0 = 298.15\,\text{K}``.  Dynamic viscosity uses
    ``\mu(T) = A\cdot10^{B/(T-C)}`` with ``A=2.414\times10^{-5}`` ``\text{Pa·s}``,
    ``B=247.8`` and ``C=140``.
    """

    V: float  # volume (m^3-ish)
    phi: float = 0.0  # electric potential (V), future use
    n: Dict[str, float] = field(default_factory=dict)  # moles by species
    pressure: float = 10
    temperature: float = 298.15
    density: float = 1000.0  # kg/m^3 for water at room temp
    viscosity: float = 1e-3  # Pa·s at room temp
    heat_capacity: float = 4181.3  # J/(kg·K) for water
    compressibility: float = 0.0  # (Pa^-1)
    min_pressure: float | None = None
    max_pressure: float | None = None
    max_flux: float | None = None  # maximum |dV| per step

    def conc(self, species: List[str]) -> dict:
        V = max(self.V, 1e-18)
        return {sp: self.n.get(sp, 0.0) / V for sp in species}

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------
    def update_temperature(self, heat: float) -> None:
        """Update temperature from an energy input.

        Parameters
        ----------
        heat:
            Energy added to the bath in joules.  Positive heats the bath.

        Uses ``ΔT = Q / (ρ c_p V)`` and updates density/viscosity using
        simple water models documented in the class notes.
        """

        mass = self.density * max(self.V, 1e-18)
        if mass > 0.0 and self.heat_capacity > 0.0:
            dT = heat / (mass * self.heat_capacity)
            self.temperature += dT

        # Update density (linear thermal expansion around 25°C)
        beta = 2.07e-4  # 1/K
        self.density = 1000.0 * (1 - beta * (self.temperature - 298.15))

        # Update dynamic viscosity via empirical relation
        A = 2.414e-5
        B = 247.8
        C = 140.0
        self.viscosity = A * 10 ** (B / (self.temperature - C))

    # ------------------------------------------------------------------
    # Compressibility helpers
    # ------------------------------------------------------------------
    def cap_fluxes(self, dV: float, dS: Dict[str, float]) -> tuple[float, Dict[str, float]]:
        """Cap proposed cell volume/solute changes by remaining compressibility.

        Parameters
        ----------
        dV:
            Proposed change in cell volume (bath receives the opposite).
        dS:
            Proposed change in moles for each species (cell perspective).

        Returns
        -------
        dV, dS:
            Possibly reduced volume and solute changes.
        """

        if self.compressibility <= 0.0:
            return dV, dS

        kV = self.compressibility * max(self.V, 1e-18)
        allowed = dV
        if dV > 0.0 and self.min_pressure is not None:
            # bath loses volume → pressure drops
            allowed = min(allowed, (self.pressure - self.min_pressure) * kV)
        if dV < 0.0 and self.max_pressure is not None:
            # bath gains volume → pressure rises
            allowed = max(allowed, (self.pressure - self.max_pressure) * kV)

        if allowed != dV and dV != 0.0:
            scale = allowed / dV
            dS = {sp: val * scale for sp, val in dS.items()}
        return allowed, dS

    # ------------------------------------------------------------------
    # Flux application
    # ------------------------------------------------------------------
    def apply_physics(self, dV: np.ndarray, dS: np.ndarray, state: Dict) -> tuple[np.ndarray, np.ndarray]:
        """Apply bath physics to proposed fluxes.

        Parameters
        ----------
        dV, dS:
            Proposed cell volume and solute fluxes (numpy arrays).
        state:
            Dictionary carrying simulation context (expects ``species`` list and
            optional ``heat`` entry).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Possibly reduced volume and solute changes.
        """

        dV = np.asarray(dV, dtype=float)
        dS = np.asarray(dS, dtype=float)

        # Global throughput cap (simple magnitude limit)
        if self.max_flux is not None:
            total = float(np.sum(np.abs(dV)))
            if total > self.max_flux:
                scale = self.max_flux / max(total, 1e-18)
                dV = dV * scale
                dS = dS * scale

        species: List[str] = state.get("species", [])

        # Compressibility/pressure bounds per cell
        if self.compressibility > 0.0 and species:
            for i in range(dV.shape[0]):
                cell_dS = {sp: dS[i, j] for j, sp in enumerate(species)}
                dV_i, dS_i = self.cap_fluxes(float(dV[i]), cell_dS)
                dV[i] = dV_i
                for j, sp in enumerate(species):
                    dS[i, j] = dS_i.get(sp, 0.0)

        # Update bath composition
        if species:
            totals = dS.sum(axis=0)
            for j, sp in enumerate(species):
                self.n[sp] = max(self.n.get(sp, 0.0) - float(totals[j]), 0.0)

        # Volume and pressure updates
        net_dV = float(np.sum(dV))
        self.V = max(self.V - net_dV, 0.0)
        if self.compressibility > 0.0 and self.V > 0.0:
            self.pressure += -(net_dV / (self.compressibility * self.V))
            if self.min_pressure is not None:
                self.pressure = max(self.pressure, self.min_pressure)
            if self.max_pressure is not None:
                self.pressure = min(self.pressure, self.max_pressure)

        # Thermal update if energy provided
        heat = state.get("heat", 0.0)
        if heat:
            self.update_temperature(float(heat))

        return dV, dS

    # ------------------------------------------------------------------
    # Diagnostics / finalisation
    # ------------------------------------------------------------------
    def finalize_step(self) -> Dict[str, float]:
        """Finalize thermodynamic state for external observers.

        Returns
        -------
        Dict[str, float]
            Mapping with ``pressure``, ``temperature`` and ``viscosity``.  The
            method performs no additional physics; it merely exposes the latest
            values so downstream viewers can verify that the bath remains within
            physical bounds.
        """

        return {
            "pressure": float(self.pressure),
            "temperature": float(self.temperature),
            "viscosity": float(self.viscosity),
        }


__all__ = ["Bath"]
