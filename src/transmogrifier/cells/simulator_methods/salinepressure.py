from sympy import symbols, sin, lambdify, Integer, Float
from math import pi, ceil, floor, isfinite

from src.transmogrifier.bitbitbuffer.helpers.cell_proposal import CellProposal
from .logutil import logger
from ..cellsim.api.saline import run_balanced_saline_sim as cs_run_balanced
from ..cell_consts import CELL_COUNT
from dataclasses import dataclass, field

@dataclass
class Organelle:
    volume_total: float                 # V_o = solid + lumen
    lumen_fraction: float = 0.7         # fraction of total that can hold solvent/solute
    # transport parameters
    Lp0: float = 0.01
    Ps0: dict = field(default_factory=lambda: {"Na":0.01, "K":0.01, "Cl":0.01, "Imp":0.0})
    sigma: dict = field(default_factory=lambda: {"Na":0.9, "K":0.9, "Cl":0.9, "Imp":1.0})
    Ea_Lp: float | None = None
    Ea_Ps: dict = field(default_factory=lambda: {"Na":None, "K":None, "Cl":None, "Imp":None})
    # lumen contents
    solute: dict = field(default_factory=lambda: {"Na":0.0, "K":0.0, "Cl":0.0, "Imp":0.0})
    # anchoring
    anchor_stiffness: float = float("inf")  # ∞ = rigidly cytoskeleton-anchored
    eps_ref: float = 0.0                    # reference strain for tether
    # derived
    def lumen_volume(self) -> float:
        return max(self.volume_total * self.lumen_fraction, 0.0)

def sphere_area_from_volume(V):
    R = (3.0 * V / (4.0 * pi)) ** (1.0/3.0)
    return 4.0 * pi * R * R, R
class SalineHydraulicSystem:
    """
    Simulates N cells in a bath with time-varying salinity and pressure.
    
    math_type='float' : continuous Euler step
    math_type='int'   : discrete volumes, choose int_method:
        - 'truncate' (default): floor(frac*width), last cell gets any remainder
        - 'adams'             : Adams ceiling + knock‑off to zero sum
    protect_under_one: if True, quotas < 1 are “too expensive to remove” (cost=∞)
    bump_under_one   : if True, any final ceiling < 1 is bumped up to 1
    """
    def __init__(self,
                 s_exprs,      # list of Sympy expressions for salinity s_i(t)
                 p_exprs,      # list of Sympy expressions for pressure p_i(t)
                 width=50,     # total bar length in characters
                 chars=None,   # list of characters for each cell
                 tau=1.0,      # relaxation time constant
                 math_type='float',   # 'float' or 'int'
                 int_method='adams',# 'truncate' or 'adams'
                 protect_under_one=True,
                 bump_under_one=False,
                 epsilon=1e-6 # small clamp for denominators
                 ):
        assert math_type in ('float','int')
        assert int_method in ('truncate','adams')
        self.math_type          = math_type
        self.int_method         = int_method
        self.protect_under_one  = protect_under_one
        self.bump_under_one     = bump_under_one
        self.width              = int(width)
        self.tau                = tau if math_type=='float' else max(1,int(tau))
        self.chars              = chars or ['#','=','-'][:len(s_exprs)]
        self.N                  = len(s_exprs)
        self.epsilon            = float(epsilon)

        # Symbolic time
        self.t = symbols('t')
        # Keep salinity & pressure expressions
        self.s_exprs = s_exprs
        self.p_exprs = p_exprs

        # Lambdify each s_i and p_i separately
        self.s_funcs = [lambdify(self.t, expr, 'math') for expr in s_exprs]
        self.p_funcs = [lambdify(self.t, expr, 'math') for expr in p_exprs]

        # initialize dynamic state
        self.reset_state()
    @staticmethod
    def run_saline_sim(sim, *, as_float=False):
        # 1) Instantiate engine with your per‐cell salinity & pressure expressions (or plain numbers)
        SalineHydraulicSystem.update_s_p_expressions(sim, sim.cells, as_float=as_float)
        sim.engine = SalineHydraulicSystem(
            sim.s_exprs,           # e.g. [Integer(s0), Integer(s1), …]
            sim.p_exprs,           # e.g. [Integer(p0), Integer(p1), …]
            width=sim.bitbuffer.mask_size, # the total bit‐space you’re dividing
            chars=[chr(97+i) for i in range(CELL_COUNT)],
            tau=5, math_type='int',
            int_method='adams',
            protect_under_one=True,
            bump_under_one=True
        )
        for cell in sim.cells:
            if cell.leftmost is None:
                #print(f"Line 67: Cell {cell.label} leftmost is None, setting to left {cell.left}")
                cell.leftmost = cell.left
            if cell.rightmost is None:
                #print(f"Line 70: Cell {cell.label} rightmost is None, setting to right - 1: {cell.right - 1}")
                cell.rightmost = cell.right - 1
        # 2) Ask for the equilibrium fractions at t=0
        sim.fractions = sim.engine.equilibrium_fracs(0.0)
        #for cell in sim.cells:
            #if cell.salinity == 0:
                #cell.salinity = 1

        necessary_size = sim.bitbuffer.intceil(sum(cell.salinity for cell in sim.cells if hasattr(cell, 'salinity') and cell.salinity > 0), sim.system_lcm)

        if sim.bitbuffer.mask_size < necessary_size:
            offsets = [sim.bitbuffer.intceil((cell.rightmost - cell.leftmost)//2+cell.leftmost, cell.stride) for cell in sim.cells if hasattr(cell, 'leftmost')]
            sizes = [(cell.salinity) for cell in sim.cells if hasattr(cell, 'salinity') and cell.salinity > 0]
            size_and_offsets = sorted(list(zip(sizes, offsets)), reverse=True, key=lambda x: x[1])
            for size, offset in size_and_offsets:
                sim.expand([offset], sim.bitbuffer.intceil(size, sim.lcm(sim.cells)), sim.cells, sim.cells)
        proposals = [CellProposal(cell) for cell in sim.cells]
        proposals = sim.snap_cell_walls(sim.cells, proposals)

        return proposals


    def run_balanced_saline_sim(self, mode="open"):
        """Balance the system then run the standard saline simulation via cellsim."""
        return cs_run_balanced(self, mode=mode)
    def reset_state(self):
        """Start at t=0 with equal volumes."""
        self.current_t = 0.0
        if self.math_type=='float':
            self.volumes = [self.width/self.N]*self.N
        else:
            # use the integer allocation method even at reset
            fracs = [1/self.N]*self.N
            self.volumes = self._integer_allocate(fracs)

    def equilibrium_fracs(self, t):
        """
        Compute equilibrium fractions safely:
          r_i = s_i/max(p_i,ε)
          sum_r = sum(r_i), fallback to uniform if too small
          frac_i = r_i/sum_r
        """
        if not hasattr(self, 's_funcs') or not hasattr(self, 'p_funcs'):
            self.t       = symbols('t')
            self.s_funcs = [lambdify(self.t, expr, 'math') for expr in self.s_exprs]
            self.p_funcs = [lambdify(self.t, expr, 'math') for expr in self.p_exprs]
        s_vals = [f(t) for f in self.s_funcs]
        p_vals = [f(t) for f in self.p_funcs]
        # safe ratios
        r_vals = []
        for si, pi in zip(s_vals, p_vals):
            denom = pi if abs(pi)>self.epsilon else self.epsilon
            r_vals.append(si/denom)
        sum_r = sum(r_vals)
        if abs(sum_r)<self.epsilon:
            return [1.0/self.N]*self.N
        return [rv/sum_r for rv in r_vals]


    def _integer_allocate(self, fracs):
        W = self.width
        if self.int_method=='truncate':
            # 1) compute raw quotas
            quotas   = [frac * W for frac in fracs]
            # 2) take ceilings
            ceilings = [ceil(q) for q in quotas]
            S        = sum(ceilings)
            # 3) how many too many
            K = S - W
            if K <= 0:
                return ceilings

            # 4) build cost list — but treat quotas < 1 as "too expensive to remove"
            costs = []
            for i, q in enumerate(quotas):
                if q <= 1.0:
                    cost = float('inf')
                else:
                    # cost = 1 − frac(q)
                    cost = 1 - (q - floor(q))
                costs.append((cost, i))

            # 5) remove from the K smallest‐cost entries
            to_remove = sorted(costs, key=lambda x: x[0])[:K]
            for _, idx in to_remove:
                ceilings[idx] -= 1

            return ceilings
 
        # --- Adams + zero‑sum ---
        # 1) raw quotas
        quotas = [frac*W for frac in fracs]
        # 2) preliminary ceilings
        if self.bump_under_one:
            ceilings = [max(1, ceil(q)) for q in quotas]
        else:
            ceilings = [ceil(q) for q in quotas]
        S = sum(ceilings)
        # 3) overshoot
        K = S - W
        if K <= 0:
            return ceilings
 
        # 4) compute removal "cost"
        #    if protect_under_one, any q<1 is too expensive to remove
        costs = []
        for i, q in enumerate(quotas):
            if q < 1.0 and self.protect_under_one:
                cost = float('inf')
            else:
                cost = 1 - (q - floor(q))
            costs.append((cost, i))
 
        # remove from those with smallest cost (largest fractional parts)
        to_remove = sorted(costs, key=lambda x: x[0])[:K]
        for _, idx in to_remove:
            ceilings[idx] -= 1
        return ceilings

    def equilibrium_bar(self, t):
        """ASCII‐bar at exact equilibrium (no dynamics)."""
        fracs = self.equilibrium_fracs(t)
        if self.math_type=='float':
            segs = [int(frac*self.width) for frac in fracs]
            segs[-1] += self.width - sum(segs)
        else:
            segs = self._integer_allocate(fracs)
        return ''.join(self.chars[i]*segs[i] for i in range(self.N))

    def step(self, dt=1.0):
        """
        One Euler‐step of dV/dt = (V_eq - V)/τ
        """
        fracs = self.equilibrium_fracs(self.current_t)
        if self.math_type=='float':
            targets = [frac*self.width for frac in fracs]
            for i in range(self.N):
                self.volumes[i] += (targets[i]-self.volumes[i])*(dt/self.tau)
        else:
            # compute integer targets by chosen method
            targets = self._integer_allocate(fracs)
            for i in range(self.N):
                delta = targets[i] - self.volumes[i]
                self.volumes[i] += (delta + 1)//self.tau

        # fix any drift
        diff = self.width - sum(self.volumes)
        self.volumes[-1] += diff

        self.current_t += dt
        
        return ''.join(self.chars[i]*int(self.volumes[i]) for i in range(self.N))

    def run_equilibrium(self, steps=20):
        for k in range(steps+1):
            tt = 2*pi*k/steps
            logger.info(f"t={tt:5.2f} → {self.equilibrium_bar(tt)}")

    def run_dynamics(self, steps=20, total_time=2*pi):
        self.reset_state()
        dt = total_time/steps
        for _ in range(steps+1):
            bar = self.step(dt)
            logger.info(f"t={self.current_t:5.2f} → {bar}")

    def balance_system(self, cells, bitbuffer,
                    mode='open',
                    C_ext=None, p_ext=None,
                    bath=None,                 # object with .volume, .solute (dict), .pressure, .temperature, .compressibility(optional)
                    dt=1e-3,                   # small base step; we adapt down as needed
                    max_steps=200000,
                    tol_vol=1e-9, tol_conc=1e-9,
                    species=("Na", "K", "Cl", "Imp"),  # Imp = impermeant anion
                    R=8.314):
        """
        Dream-grade mechano–osmotic balance with conservation and feedbacks.

        Water:  Jv = Lp(A,T) * A * ( (P_ext - P_i - ΔP_tension) - Σ_i σ_i RT φ_i (C_ext_i - C_i) )
        Solute: Js_i = Ps_i(A,T) * A * (C_ext_i - C_i) + (1-σ_i) * C_i * Jv    # solvent drag

        Mechanics:
        A(V) from sphere: R = (3V/4π)^(1/3),  A = 4πR^2
        strain ε = A/A0 - 1
        Tension T = k_s * ε + η_s * dε/dt
        ΔP_tension = 2T / R   (Laplace)

        Active transport (optional):
        Na/K pump: Jpump_Na = -3*J_pump, Jpump_K = +2*J_pump (A·conc/time), depends on ATP/Tension.

        Bath:
        Finite reservoir; updates volume/solute by -Σ Δ; optional compressibility for pressure updates.
        """
        if mode != 'open' or not cells:
            return [CellProposal(c) for c in cells]

        # ---- Utilities (no external deps) ----------------------------------------
        from math import pi

        def sphere_R(V):  # radius from volume
            return (3.0*V/(4.0*pi))**(1.0/3.0)

        def sphere_A(V):
            Rv = sphere_R(V)
            return 4.0*pi*Rv*Rv, Rv

        def arrhenius(P0, Ea, T):
            # If Ea not supplied, return P0
            return P0 if Ea is None else P0 * (2.718281828)**(-Ea/(R*T))

        # ---- Initialise missing fields ------------------------------------------
        # One-time default wiring if upstream hasn’t set these yet.
        for c in cells:
            if not hasattr(c, "volume"):
                c.volume = float(c.right - c.left)
            if not hasattr(c, "initial_volume"):
                c.initial_volume = float(c.volume)
            if not hasattr(c, "A0"):
                c.A0, _ = sphere_A(c.initial_volume)
            if not hasattr(c, "base_pressure"):
                c.base_pressure = float(getattr(c, "pressure", 0.0))
            if not hasattr(c, "elastic_k"):
                c.elastic_k = 0.1
            if not hasattr(c, "visc_eta"):
                c.visc_eta = 0.0  # set >0 for viscoelastic damping
            if not hasattr(c, "Lp0"):
                # fall back to wall permeabilities if present
                lp_l = float(getattr(c, "l_solvent_permiability", 1.0))
                lp_r = float(getattr(c, "r_solvent_permiability", 1.0))
                c.Lp0 = 0.5*(lp_l + lp_r)
            if not hasattr(c, "sigma"):
                c.sigma = {sp: (1.0 if sp in ("Imp",) else 0.9) for sp in species}
            if not hasattr(c, "Ps0"):
                c.Ps0 = {sp: (0.0 if sp == "Imp" else 0.01) for sp in species}
            if not hasattr(c, "Ea_Ps"):
                c.Ea_Ps = {sp: None for sp in species}
            if not hasattr(c, "Ea_Lp"):
                c.Ea_Lp = None
            if not hasattr(c, "solute"):
                # Map legacy salinity into one bucket if needed
                S = float(getattr(c, "salinity", 0.0))
                c.solute = {sp: (S if sp == "Imp" else 0.0) for sp in species}
            # initialize organelles list
            if not hasattr(c, "organelles"):
                c.organelles = []

            # keep previous strain for viscous term
            c._prev_eps = 0.0

        # Bath defaults
        if bath is None:
            # Finite bath equal to total cell volume, with given C_ext for a single lumped species
            V_bath = sum(c.volume for c in cells)
            T_bath = float(getattr(self, "temperature", 298.15))
            P_bath = float(p_ext if p_ext is not None else getattr(self, "external_pressure", 1e4))
            # If caller provided a single C_ext earlier, put it in Na for demo
            Cext_single = float(C_ext if C_ext is not None else getattr(self, "external_concentration", 1500.0))
            S_bath = {sp: (Cext_single*V_bath if sp == "Na" else 0.0) for sp in species}
            bath = type("Bath", (), {})()
            bath.volume = V_bath
            bath.temperature = T_bath
            bath.pressure = P_bath
            bath.solute = S_bath
            bath.compressibility = 0.0  # 0 = fixed pressure
        else:
            # sanity fills
            for sp in species:
                bath.solute.setdefault(sp, 0.0)
            if not hasattr(bath, "compressibility"):
                bath.compressibility = 0.0

        # ---- Organelle utilities: free volume and exchange step ----
        def cytosol_free_volume(c):
            occ = sum(o.volume_total for o in getattr(c, "organelles", []))
            return max(c.volume - occ, 1e-18)

        def organelle_exchange_step(c, T_bath, dt):
            """One explicit inner step: cytosol ↔ each organelle lumen."""
            V_free = cytosol_free_volume(c)
            Ccyt = {sp: c.solute[sp] / V_free for sp in species}
            for o in getattr(c, "organelles", []):
                V_lum = max(o.lumen_volume(), 1e-18)
                A_o, R_o = sphere_area_from_volume(V_lum)
                Corg = {sp: o.solute[sp] / V_lum for sp in o.solute.keys()}
                A_cell, R_cell = sphere_A(c.volume)
                eps_cell = max(A_cell / c.A0 - 1.0, 0.0)
                Lp = arrhenius(o.Lp0, o.Ea_Lp, T_bath) * (1.0 + 0.3 * eps_cell)
                osm = 0.0
                for sp in o.sigma.keys():
                    osm += o.sigma[sp] * R * T_bath * (Ccyt[sp] - Corg[sp])
                dP = 0.0
                Jv = Lp * A_o * (dP - osm)
                dV = Jv * dt
                dS = {}
                for sp in o.Ps0.keys():
                    Ps = arrhenius(o.Ps0[sp], o.Ea_Ps.get(sp), T_bath) * (1.0 + 0.5 * eps_cell)
                    Js = Ps * A_o * (Ccyt[sp] - Corg[sp]) + (1.0 - o.sigma[sp]) * Corg[sp] * Jv
                    dS[sp] = Js * dt
                o.volume_total = max(o.volume_total + dV, 1e-18)
                for sp in dS.keys():
                    o.solute[sp] = max(o.solute[sp] + dS[sp], 0.0)
                    c.solute[sp] = max(c.solute[sp] - dS[sp], 0.0)

        # ---- Integrate -----------------------------------------------------------
        proposals = [CellProposal(c) for c in cells]

        t = 0.0
        for step in range(max_steps):
            T_bath = bath.temperature
            # Concentrations
            Cext = {sp: (bath.solute[sp]/bath.volume if bath.volume > 0 else 0.0) for sp in species}

            max_rel = 0.0
            sum_dV = 0.0

            for c in cells:
                # Geometry, strain and tension
                A, Rv = sphere_A(c.volume)
                eps = (A / c.A0) - 1.0
                deps_dt = (eps - c._prev_eps) / dt
                Tension = c.elastic_k * eps + c.visc_eta * deps_dt
                dP_tension = (2.0 * Tension / max(Rv, 1e-12))  # Laplace
                c._prev_eps = eps

                # Internal hydrostatic pressure (turgor + base)
                P_i = c.base_pressure + dP_tension

                # Osmotic term: Σ σ_i RT (Cext_i - C_i)
                Cint = {}
                for sp in species:
                    Cint[sp] = c.solute[sp] / max(c.volume, 1e-18)
                osm_term = 0.0
                for sp in species:
                    sigma_i = c.sigma.get(sp, 1.0)
                    osm_term += sigma_i * R * T_bath * (Cext[sp] - Cint[sp])

                # Permeabilities (tension & temperature dependence)
                Lp = arrhenius(c.Lp0, c.Ea_Lp, T_bath) * (1.0 + 0.3*max(eps, 0.0))  # mild tension boost
                dP = bath.pressure - P_i
                Jv = Lp * A * (dP - osm_term)                # m^3/s (units relative)
                dV = Jv * dt

                # Solute fluxes (with solvent drag)
                dS = {}
                for sp in species:
                    Ps0 = c.Ps0.get(sp, 0.0)
                    Ps = arrhenius(Ps0, c.Ea_Ps.get(sp), T_bath) * (1.0 + 0.5*max(eps, 0.0))
                    Cavg = 0.5*(Cext[sp] + Cint[sp])
                    sigma_i = c.sigma.get(sp, 1.0)
                    Js = Ps * A * (Cext[sp] - Cint[sp]) + (1.0 - sigma_i) * Cint[sp] * Jv
                    dS[sp] = Js * dt

                # (Optional) Na/K pump (tiny stabiliser; set to 0 to disable)
                J_pump = getattr(c, "J_pump", 0.0)  # mol/s across membrane
                if J_pump:
                    dS["Na"] = dS.get("Na", 0.0) - 3.0 * J_pump * dt
                    dS["K"]  = dS.get("K", 0.0)  + 2.0 * J_pump * dt

                # Update cell and bath, enforce positivity
                c.volume = max(c.volume + dV, 1e-18)
                for sp in species:
                    c.solute[sp] = max(c.solute[sp] + dS[sp], 0.0)

                bath.volume = max(bath.volume - dV, 1e-18)
                for sp in species:
                    bath.solute[sp] = max(bath.solute[sp] - dS[sp], 0.0)

                # Relative step size for adaptive dt
                rel = abs(dV) / max(c.volume, 1e-18)
                if rel > max_rel:
                    max_rel = rel
                sum_dV += dV

            # Bath pressure model (optional compressibility)
            if bath.compressibility and bath.compressibility > 0.0:
                # dP = -K * dV / V  (very rough bulk modulus-like)
                bath.pressure += -bath.compressibility * (sum_dV / max(bath.volume, 1e-18))

            # Convergence check
            if max_rel < tol_vol:
                # also check concentration change
                max_dc = 0.0
                for c in cells:
                    for sp in species:
                        Ci = c.solute[sp]/max(c.volume,1e-18)
                        Ce = bath.solute[sp]/max(bath.volume,1e-18)
                        max_dc = max(max_dc, abs(Ce - Ci))
                if max_dc < tol_conc:
                    break

            # Adaptive dt to keep |ΔV|/V small (explicit stability)
            if max_rel > 1e-3:
                dt *= 0.5
            elif max_rel < 1e-5:
                dt *= 1.1

            t += dt

        # Map net expansion into bitbuffer event (keep your alignment logic)
        proposals = [CellProposal(c) for c in cells]
        if sum_dV > 0 and isfinite(sum_dV):
            system_lcm = self.lcm(cells)
            delta_bits = bitbuffer.intceil(bitbuffer.round(sum_dV), system_lcm)
            if 0 < delta_bits < 10**12:
                manual_event = (None, bitbuffer.mask_size, delta_bits)
                proposals = bitbuffer.expand([manual_event], cells, proposals)

        # For downstream visualisations, keep convenient scalars
        for c in cells:
            # Use membrane tension from geometric strain to derive pressure
            A_curr, R_curr = sphere_A(c.volume)
            strain = max(A_curr / c.A0 - 1.0, 0.0)
            c.pressure = c.base_pressure + (2.0 * (c.elastic_k * strain) / max(R_curr, 1e-12))
            conc_dict = {sp: c.solute[sp]/c.volume for sp in species}
            c.concentration = conc_dict.get('Imp', 0.0)
            c.concentrations = conc_dict

        return proposals
    @staticmethod
    def update_s_p_expressions(sim, cells, *, as_float=False):
        if as_float:
            sim.s_exprs = [Float(cell.salinity) for cell in cells]
            sim.p_exprs = [Float(cell.pressure) for cell in cells]
        else:
            sim.s_exprs = [Integer(cell.salinity) for cell in cells]
            sim.p_exprs = [Integer(cell.pressure) for cell in cells]

import time
if __name__ == '__main__':
    # ASCII animation demo: vary salinity and pressure sinusoidally with distinct
    # amplitudes, frequencies, and phases per cell and property.
    try:
        import shutil
    except Exception:
        shutil = None

    # --- Config ---
    n = 10  # number of cells
    t = symbols('t')

    # Vary per-cell with different amplitude, frequency, and phase
    # Keep amplitudes <= 0.9 to stay positive since baseline is 1.0
    s_exprs = []
    p_exprs = []
    for i in range(n):
        amp_s = 0.55 + 0.35*((i % 3)/2)           # 0.55, 0.725, 0.9
        amp_p = 0.35 + 0.25*((i % 4)/3)           # 0.35..0.6
        freq_s = 1.0 + 0.2*((i % 3) - 1)          # 0.8, 1.0, 1.2
        freq_p = 1.5 + 0.15*((i % 4) - 1.5)       # 1.275 .. 1.725
        phase_s = 2*pi*i/n                        # distributed around circle
        phase_p = pi/4 + pi*i/(n//2 if n>1 else 1)  # staggered differently
        s_exprs.append(1 + amp_s * sin(freq_s*t + phase_s))
        p_exprs.append(1 + amp_p * sin(freq_p*t + phase_p))

    # Terminal width or fallback
    term_width = 100
    if shutil is not None:
        try:
            term_width = max(40, min(160, (shutil.get_terminal_size().columns or 100)))
        except Exception:
            pass

    # Instantiate system
    sys_int = SalineHydraulicSystem(
        s_exprs, p_exprs,
        width=term_width - 2,  # leave padding
        chars=[chr(97+i) for i in range(n)],
        tau=5, math_type='float',
        int_method='adams',
        protect_under_one=True,
        bump_under_one=True
    )

    # Animation params
    sys_int.reset_state()
    dt = (2 * pi) / 60  # ~60 steps per fundamental cycle
    fps = 20
    delay = 1.0 / fps

    # Precompile lambdas for readout (already lambdified in __init__)
    s_funcs = sys_int.s_funcs
    p_funcs = sys_int.p_funcs

    # ANSI helpers
    def hide_cursor():
        print('\x1b[?25l', end='', flush=True)

    def show_cursor():
        print('\x1b[?25h', end='', flush=True)

    def clear_to_top():
        # Clear screen and move cursor home
        print('\x1b[2J\x1b[H', end='')

    # Render
    try:
        hide_cursor()
        start = time.time()
        while True:
            # Step simulation and build bar
            bar = sys_int.step(dt)

            # Sample current s, p for readout at this time
            tt = sys_int.current_t
            s_vals = [f(tt) for f in s_funcs]
            p_vals = [f(tt) for f in p_funcs]

            # Frame
            clear_to_top()
            title = f"SalineHydraulicSystem demo  t={tt:5.2f}  math={sys_int.math_type}  (Ctrl+C to exit)"
            print(title)
            print('|' + bar + '|')

            # Compact per-cell stats line (wrap if needed)
            lines = []
            line = []
            for i, (sv, pv) in enumerate(zip(s_vals, p_vals)):
                seg = f"{sys_int.chars[i]}:s={sv:4.2f} p={pv:4.2f}"
                if sum(len(x)+2 for x in line) + len(seg) + 2 > term_width:
                    lines.append('  '.join(line))
                    line = [seg]
                else:
                    line.append(seg)
            if line:
                lines.append('  '.join(line))
            for ln in lines:
                print(ln)

            # Pace to target FPS
            elapsed = time.time() - start
            # Keep it simple: constant delay
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        print()  # move to next line for clean prompt