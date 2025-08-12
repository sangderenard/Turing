from sympy import symbols, sin, lambdify, Integer, Float
from math import pi, ceil, floor, isfinite

from src.transmogrifier.bitbitbuffer.helpers.cell_proposal import CellProposal
from .logutil import logger
from ..cell_sim import CellSim
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
                cell.leftmost = cell.left
            if cell.rightmost is None:
                cell.rightmost = cell.right - 1
        sim.fractions = sim.engine.equilibrium_fracs(0.0)

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
        """Balance the system then run the standard saline simulation."""
        wrapped = [CellSim.get(cell) for cell in self.cells]
        for w in wrapped:
            w.pull_from_cell()

        proposals = self.balance_system(wrapped, self.bitbuffer, mode)

        for w in wrapped:
            w.push_to_cell()

        unwrapped = []
        for w, prop in zip(wrapped, proposals):
            new_prop = CellProposal(w.cell)
            new_prop.left = prop.left
            new_prop.right = prop.right
            new_prop.leftmost = prop.leftmost
            new_prop.rightmost = prop.rightmost
            new_prop.salinity = getattr(prop, "salinity", getattr(w.cell, "salinity", 0))
            new_prop.pressure = getattr(prop, "pressure", getattr(w.cell, "pressure", 0))
            unwrapped.append(new_prop)

        for cell, proposal in zip(self.cells, unwrapped):
            cell.apply_proposal(proposal)

        proposals = self.run_saline_sim(as_float=True)
        return proposals
    def reset_state(self):
        """Start at t=0 with equal volumes."""
        self.current_t = 0.0
        if self.math_type=='float':
            self.volumes = [self.width/self.N]*self.N
        else:
            fracs = [1/self.N]*self.N
            self.volumes = self._integer_allocate(fracs)

    def equilibrium_fracs(self, t):
        if not hasattr(self, 's_funcs') or not hasattr(self, 'p_funcs'):
            self.t       = symbols('t')
            self.s_funcs = [lambdify(self.t, expr, 'math') for expr in self.s_exprs]
            self.p_funcs = [lambdify(self.t, expr, 'math') for expr in self.p_exprs]
        s_vals = [f(t) for f in self.s_funcs]
        p_vals = [f(t) for f in self.p_funcs]
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
            quotas   = [frac * W for frac in fracs]
            ceilings = [ceil(q) for q in quotas]
            S        = sum(ceilings)
            K = S - W
            if K <= 0:
                return ceilings

            costs = []
            for i, q in enumerate(quotas):
                if q <= 1.0:
                    cost = float('inf')
                else:
                    cost = 1 - (q - floor(q))
                costs.append((cost, i))

            to_remove = sorted(costs, key=lambda x: x[0])[:K]
            for _, idx in to_remove:
                ceilings[idx] -= 1

            return ceilings
 
        quotas = [frac*W for frac in fracs]
        if self.bump_under_one:
            ceilings = [max(1, ceil(q)) for q in quotas]
        else:
            ceilings = [ceil(q) for q in quotas]
        S = sum(ceilings)
        K = S - W
        if K <= 0:
            return ceilings
 
        costs = []
        for i, q in enumerate(quotas):
            if q < 1.0 and self.protect_under_one:
                cost = float('inf')
            else:
                cost = 1 - (q - floor(q))
            costs.append((cost, i))
 
        to_remove = sorted(costs, key=lambda x: x[0])[:K]
        for _, idx in to_remove:
            ceilings[idx] -= 1
        return ceilings

    def equilibrium_bar(self, t):
        fracs = self.equilibrium_fracs(t)
        if self.math_type=='float':
            segs = [int(frac*self.width) for frac in fracs]
            segs[-1] += self.width - sum(segs)
        else:
            segs = self._integer_allocate(fracs)
        return ''.join(self.chars[i]*segs[i] for i in range(self.N))

    def step(self, dt=1.0):
        fracs = self.equilibrium_fracs(self.current_t)
        if self.math_type=='float':
            targets = [frac*self.width for frac in fracs]
            for i in range(self.N):
                self.volumes[i] += (targets[i]-self.volumes[i])*(dt/self.tau)
        else:
            targets = self._integer_allocate(fracs)
            for i in range(self.N):
                delta = targets[i] - self.volumes[i]
                self.volumes[i] += (delta + 1)//self.tau

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
        if mode != 'open' or not cells:
            return [CellProposal(c) for c in cells]

        from math import pi

        def sphere_R(V):
            return (3.0*V/(4.0*pi))**(1.0/3.0)

        def sphere_A(V):
            Rv = sphere_R(V)
            return 4.0*pi*Rv*Rv, Rv

        def arrhenius(P0, Ea, T):
            return P0 if Ea is None else P0 * (2.718281828)**(-Ea/(R*T))

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
                c.visc_eta = 0.0
            if not hasattr(c, "Lp0"):
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
                S = float(getattr(c, "salinity", 0.0))
                c.solute = {sp: (S if sp == "Imp" else 0.0) for sp in species}
            if not hasattr(c, "organelles"):
                c.organelles = []
            c._prev_eps = 0.0

        if bath is None:
            V_bath = sum(c.volume for c in cells)
            T_bath = float(getattr(self, "temperature", 298.15))
            P_bath = float(p_ext if p_ext is not None else getattr(self, "external_pressure", 1e4))
            Cext_single = float(C_ext if C_ext is not None else getattr(self, "external_concentration", 1500.0))
            S_bath = {sp: (Cext_single*V_bath if sp == "Na" else 0.0) for sp in species}
            bath = type("Bath", (), {})()
            bath.volume = V_bath
            bath.temperature = T_bath
            bath.pressure = P_bath
            bath.solute = S_bath
            bath.compressibility = 0.0
        else:
            for sp in species:
                bath.solute.setdefault(sp, 0.0)
            if not hasattr(bath, "compressibility"):
                bath.compressibility = 0.0

        def cytosol_free_volume(c):
            occ = sum(o.volume_total for o in getattr(c, "organelles", []))
            return max(c.volume - occ, 1e-18)

        def organelle_exchange_step(c, T_bath, dt):
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

        proposals = [CellProposal(c) for c in cells]

        t = 0.0
        for step in range(max_steps):
            T_bath = bath.temperature
            Cext = {sp: (bath.solute[sp]/bath.volume if bath.volume > 0 else 0.0) for sp in species}

            max_rel = 0.0
            sum_dV = 0.0

            for c in cells:
                A, Rv = sphere_A(c.volume)
                eps = (A / c.A0) - 1.0
                deps_dt = (eps - c._prev_eps) / dt
                Tension = c.elastic_k * eps + c.visc_eta * deps_dt
                dP_tension = (2.0 * Tension / max(Rv, 1e-12))
                c._prev_eps = eps

                P_i = c.base_pressure + dP_tension

                Cint = {}
                for sp in species:
                    Cint[sp] = c.solute[sp] / max(c.volume, 1e-18)
                osm_term = 0.0
                for sp in species:
                    sigma_i = c.sigma.get(sp, 1.0)
                    osm_term += sigma_i * R * T_bath * (Cext[sp] - Cint[sp])

                Lp = arrhenius(c.Lp0, c.Ea_Lp, T_bath) * (1.0 + 0.3*max(eps, 0.0))
                dP = bath.pressure - P_i
                Jv = Lp * A * (dP - osm_term)
                dV = Jv * dt

                dS = {}
                for sp in species:
                    Ps0 = c.Ps0.get(sp, 0.0)
                    Ps = arrhenius(Ps0, c.Ea_Ps.get(sp), T_bath) * (1.0 + 0.5*max(eps, 0.0))
                    Cavg = 0.5*(Cext[sp] + Cint[sp])
                    sigma_i = c.sigma.get(sp, 1.0)
                    Js = Ps * A * (Cext[sp] - Cint[sp]) + (1.0 - sigma_i) * Cint[sp] * Jv
                    dS[sp] = Js * dt

                J_pump = getattr(c, "J_pump", 0.0)
                if J_pump:
                    dS["Na"] = dS.get("Na", 0.0) - 3.0 * J_pump * dt
                    dS["K"]  = dS.get("K", 0.0)  + 2.0 * J_pump * dt

                c.volume = max(c.volume + dV, 1e-18)
                for sp in species:
                    c.solute[sp] = max(c.solute[sp] + dS[sp], 0.0)

                bath.volume = max(bath.volume - dV, 1e-18)
                for sp in species:
                    bath.solute[sp] = max(bath.solute[sp] - dS[sp], 0.0)

                rel = abs(dV) / max(c.volume, 1e-18)
                if rel > max_rel:
                    max_rel = rel
                sum_dV += dV

            if bath.compressibility and bath.compressibility > 0.0:
                bath.pressure += -bath.compressibility * (sum_dV / max(bath.volume, 1e-18))

            if max_rel < tol_vol:
                max_dc = 0.0
                for c in cells:
                    for sp in species:
                        Ci = c.solute[sp]/max(c.volume,1e-18)
                        Ce = bath.solute[sp]/max(bath.volume,1e-18)
                        max_dc = max(max_dc, abs(Ce - Ci))
                if max_dc < tol_conc:
                    break

            if max_rel > 1e-3:
                dt *= 0.5
            elif max_rel < 1e-5:
                dt *= 1.1

            t += dt

        proposals = [CellProposal(c) for c in cells]
        if sum_dV > 0 and isfinite(sum_dV):
            system_lcm = self.lcm(cells)
            delta_bits = bitbuffer.intceil(bitbuffer.round(sum_dV), system_lcm)
            if 0 < delta_bits < 10**12:
                manual_event = (None, bitbuffer.mask_size, delta_bits)
                proposals = bitbuffer.expand([manual_event], cells, proposals)

        for c in cells:
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
    try:
        import shutil
    except Exception:
        shutil = None

    n = 10
    t = symbols('t')

    s_exprs = []
    p_exprs = []
    for i in range(n):
        amp_s = 0.55 + 0.35*((i % 3)/2)
        amp_p = 0.35 + 0.25*((i % 4)/3)
        freq_s = 1.0 + 0.2*((i % 3) - 1)
        freq_p = 1.5 + 0.15*((i % 4) - 1.5)
        phase_s = 2*pi*i/n
        phase_p = pi/4 + pi*i/(n//2 if n>1 else 1)
        s_exprs.append(1 + amp_s * sin(freq_s*t + phase_s))
        p_exprs.append(1 + amp_p * sin(freq_p*t + phase_p))

    term_width = 100
    if shutil is not None:
        try:
            term_width = max(40, min(160, (shutil.get_terminal_size().columns or 100)))
        except Exception:
            pass

    sys_int = SalineHydraulicSystem(
        s_exprs, p_exprs,
        width=term_width - 2,
        chars=[chr(97+i) for i in range(n)],
        tau=5, math_type='float',
        int_method='adams',
        protect_under_one=True,
        bump_under_one=True
    )

    sys_int.reset_state()
    dt = (2 * pi) / 60
    fps = 20
    delay = 1.0 / fps

    s_funcs = sys_int.s_funcs
    p_funcs = sys_int.p_funcs

    def hide_cursor():
        print('\x1b[?25l', end='', flush=True)

    def show_cursor():
        print('\x1b[?25h', end='', flush=True)

    def clear_to_top():
        print('\x1b[2J\x1b[H', end='')

    try:
        hide_cursor()
        start = time.time()
        while True:
            bar = sys_int.step(dt)
            tt = sys_int.current_t
            s_vals = [f(tt) for f in s_funcs]
            p_vals = [f(tt) for f in p_funcs]
            clear_to_top()
            title = f"SalineHydraulicSystem demo  t={tt:5.2f}  math={sys_int.math_type}  (Ctrl+C to exit)"
            print(title)
            print('|' + bar + '|')

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

            elapsed = time.time() - start
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        print()
