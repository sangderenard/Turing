from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple
import uuid
import weakref
import math

# ---- Minimal "Bath" so the wrapper has a counterpart -------------------------
@dataclass
class Bath:
    volume: float
    pressure: float           # Pa (or your internal “pressure units”)
    temperature: float        # K
    solute: Dict[str, float]  # mol per species
    compressibility: float = 0.0


# ---- Cell wrapper with registry keyed by cell.label --------------------------
class CellSim:
    """
    Wraps a data Cell (your object) with simulation parameters/state.
    Reused via CellSim.get(cell): one wrapper per cell.label.
    """

    # ====== Class-level (static) defaults you can tune globally ======
    # Globals/physics
    R: float = 8.314  # J/mol/K
    species: Tuple[str, ...] = ("Na", "K", "Cl", "Imp")

    # Default permeabilities
    default_Lp0: float = 1.0          # water permeability
    default_Ps0: Dict[str, float] = { # solute permeability per species
        "Na": 0.01, "K": 0.01, "Cl": 0.02, "Imp": 0.0
    }
    default_sigma: Dict[str, float] = {# reflection coefficient
        "Na": 0.9, "K": 0.9, "Cl": 0.9, "Imp": 1.0
    }

    # Arrhenius (optional); set to None for off
    default_Ea_Lp: Optional[float] = None
    default_Ea_Ps: Dict[str, Optional[float]] = {sp: None for sp in ("Na","K","Cl","Imp")}

    # Wall mechanics: Kelvin–Voigt on membrane area strain
    default_elastic_k: float = 0.1    # N/m per strain
    default_visc_eta: float = 0.0     # N·s/m per strain rate

    # Tension→permeability modulation (soft default)
    lp_tension_boost: float = 0.3
    ps_tension_boost: float = 0.5

    # Registry: label -> weak wrapper (so gc is clean)
    _registry: "weakref.WeakValueDictionary[str, CellSim]" = weakref.WeakValueDictionary()

    # ====== Factory / registry ======
    @classmethod
    def get(cls, cell) -> "CellSim":
        """Return the wrapper for this cell.label, creating if absent."""
        label = getattr(cell, "label", None)
        if not label:
            # make a stable label once and attach it
            label = f"cell_{uuid.uuid4().hex[:12]}"
            setattr(cell, "label", label)

        inst = cls._registry.get(label)
        if inst is None:
            inst = cls(_CellRef(cell))
            cls._registry[label] = inst
        else:
            inst._cell = _CellRef(cell)  # refresh target if object was re-created
        return inst

    @classmethod
    def set_global_defaults(cls, *,
                            R: Optional[float]=None,
                            species: Optional[Iterable[str]]=None,
                            Lp0: Optional[float]=None,
                            Ps0: Optional[Dict[str,float]]=None,
                            sigma: Optional[Dict[str,float]]=None,
                            elastic_k: Optional[float]=None,
                            visc_eta: Optional[float]=None):
        if R is not None: cls.R = R
        if species is not None: cls.species = tuple(species)
        if Lp0 is not None: cls.default_Lp0 = Lp0
        if Ps0 is not None: cls.default_Ps0.update(Ps0)
        if sigma is not None: cls.default_sigma.update(sigma)
        if elastic_k is not None: cls.default_elastic_k = elastic_k
        if visc_eta is not None: cls.default_visc_eta = visc_eta

    # ====== Instance ======
    def __init__(self, cellref: "_CellRef"):
        self._cell = cellref

        # Pull initial geometry/state from the cell
        c = self.cell
        self.label: str = getattr(c, "label")
        self.volume: float = float(getattr(c, "volume", c.right - c.left))
        self.initial_volume: float = float(getattr(c, "initial_volume", self.volume))
        self.A0, _ = self._area_radius(self.initial_volume)

        # solute amounts (mol); map legacy salinity into impermeant by default
        if hasattr(c, "solute"):
            raw = dict(c.solute)
        else:
            raw = {sp: 0.0 for sp in self.species}
            raw["Imp"] = float(getattr(c, "salinity", 0.0))
        # ensure all species exist
        for sp in self.species:
            raw.setdefault(sp, 0.0)
        self.solute: Dict[str,float] = raw

        # mechanics/permeabilities (cell-level overrides if present)
        self.base_pressure: float = float(getattr(c, "base_pressure", getattr(c, "pressure", 0.0)))
        self.elastic_k: float = float(getattr(c, "elastic_k", self.default_elastic_k))
        self.visc_eta: float = float(getattr(c, "visc_eta", self.default_visc_eta))
        self.Lp0: float = float(getattr(c, "Lp0", self.default_Lp0))
        self.Ea_Lp: Optional[float] = getattr(c, "Ea_Lp", self.default_Ea_Lp)

        self.Ps0: Dict[str,float] = {sp: float(getattr(c, "Ps0", {}).get(sp, self.default_Ps0[sp]))
                                     for sp in self.species}
        self.Ea_Ps: Dict[str,Optional[float]] = {sp: getattr(c, "Ea_Ps", {}).get(sp, self.default_Ea_Ps[sp])
                                                 for sp in self.species}
        self.sigma: Dict[str,float] = {sp: float(getattr(c, "sigma", {}).get(sp, self.default_sigma[sp]))
                                       for sp in self.species}

        # optional pump rate (mol/s)
        self.J_pump: float = float(getattr(c, "J_pump", 0.0))

        # integration helpers
        self._prev_eps: float = 0.0

    # ----- Lightweight access to the live cell -----
    @property
    def cell(self):
        c = self._cell()
        if c is None:
            raise ReferenceError(f"CellSim[{self.label}] underlying cell was GC’d or lost.")
        return c

    # Delegate attribute access to the underlying cell for any fields the
    # wrapper does not override.  This lets a CellSim instance act like the
    # original cell when passed into existing algorithms that expect a plain
    # cell object (e.g. `balance_system`).
    def __getattr__(self, name):
        return getattr(self.cell, name)

    # ----- Synchronisation -----
    def pull_from_cell(self) -> None:
        """Refresh geometry/basic fields from cell (after proposals/expansion)."""
        c = self.cell
        V = float(getattr(c, "volume", c.right - c.left))
        if not hasattr(c, "initial_volume"):
            c.initial_volume = V
        self.volume = max(V, 1e-18)
        self.initial_volume = float(getattr(c, "initial_volume", self.initial_volume))
        self.A0, _ = self._area_radius(self.initial_volume)
        # allow external edits of per-cell params
        self.base_pressure = float(getattr(c, "base_pressure", self.base_pressure))
        self.elastic_k     = float(getattr(c, "elastic_k", self.elastic_k))
        self.visc_eta      = float(getattr(c, "visc_eta", self.visc_eta))
        self.Lp0           = float(getattr(c, "Lp0", self.Lp0))
        # optional solute refresh if user modified
        if hasattr(c, "solute"):
            for sp in self.species:
                self.solute[sp] = float(c.solute.get(sp, self.solute.get(sp, 0.0)))

    def push_to_cell(self) -> None:
        """Write back convenient scalars for downstream code/visuals."""
        c = self.cell
        c.volume = self.volume
        # expose pressure and per-species concentration for debugging
        pressure = self.base_pressure + self._dP_tension(self.volume)[0]
        setattr(c, "pressure", pressure)
        conc = {sp: self.solute[sp]/self.volume for sp in self.species}
        setattr(c, "concentration", conc)

    # ----- Geometry/physics helpers -----
    @staticmethod
    def _area_radius(V: float) -> Tuple[float, float]:
        R = (3.0*V/(4.0*math.pi))**(1.0/3.0)
        return 4.0*math.pi*R*R, R

    def _arrhenius(self, P0: float, Ea: Optional[float], T: float) -> float:
        if Ea is None: return P0
        # use exp(-Ea/RT) without importing numpy
        return P0 * math.exp(-Ea/(self.R*T))

    def _dP_tension(self, V: float, dt: Optional[float]=None) -> Tuple[float, float, float]:
        """Return (ΔP_tension, strain eps, d_eps_dt)"""
        A, R = self._area_radius(V)
        eps = (A/self.A0) - 1.0
        if dt is None:
            deps_dt = 0.0
        else:
            deps_dt = (eps - self._prev_eps) / dt
        Tension = self.elastic_k * eps + self.visc_eta * deps_dt
        dP = 2.0*Tension/max(R, 1e-12)
        return dP, eps, deps_dt

    # ----- One explicit sub-step for this cell against a Bath -----
    def step(self, bath: Bath, dt: float) -> Tuple[float, Dict[str,float]]:
        """
        Advance the wrapper by dt using KK/Starling with solvent drag.
        Returns (dV, dS_per_species) (positive into cell).
        """
        T = bath.temperature
        # concentrations
        Cext = {sp: bath.solute[sp]/bath.volume for sp in self.species}
        Cint = {sp: self.solute[sp]/self.volume for sp in self.species}

        # mechanics & permeabilities
        dP_tension, eps, deps_dt = self._dP_tension(self.volume, dt)
        Lp = self._arrhenius(self.Lp0, self.Ea_Lp, T) * (1.0 + self.lp_tension_boost*max(eps, 0.0))
        A, _ = self._area_radius(self.volume)

        # hydrostatic and osmotic terms
        P_i = self.base_pressure + dP_tension
        dP  = bath.pressure - P_i
        osm = 0.0
        for sp in self.species:
            osm += self.sigma[sp] * self.R * T * (Cext[sp] - Cint[sp])

        Jv = Lp * A * (dP - osm)         # volume flux
        dV = Jv * dt

        dS: Dict[str,float] = {}
        for sp in self.species:
            Ps = self._arrhenius(self.Ps0[sp], self.Ea_Ps[sp], T) * (1.0 + self.ps_tension_boost*max(eps, 0.0))
            Js = Ps * A * (Cext[sp] - Cint[sp]) + (1.0 - self.sigma[sp]) * Cint[sp] * Jv
            dS[sp] = Js * dt

        # pump (optional)
        if self.J_pump:
            dS["Na"] += -3.0*self.J_pump*dt
            dS["K"]  +=  2.0*self.J_pump*dt

        # update state, clamp positive
        self.volume = max(self.volume + dV, 1e-18)
        for sp in self.species:
            self.solute[sp] = max(self.solute[sp] + dS[sp], 0.0)

        # store strain for viscous term
        self._prev_eps = eps
        return dV, dS


# ---- lightweight weakref to underlying cell ---------------------------------
class _CellRef:
    def __init__(self, obj):
        self._ref = weakref.ref(obj)
    def __call__(self):
        return self._ref()
