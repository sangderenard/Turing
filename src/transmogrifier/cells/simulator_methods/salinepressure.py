from sympy import symbols, sin, lambdify, Integer, Float
from math import pi, ceil, floor

from src.transmogrifier.bitbitbuffer.helpers.cell_proposal import CellProposal
from ..cell_consts import CELL_COUNT
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
        """Balance the system then run the standard saline simulation."""
        proposals = self.balance_system(self.cells, self.bitbuffer, mode)
        for cell, proposal in zip(self.cells, proposals):
            cell.apply_proposal(proposal)
        proposals = self.run_saline_sim(as_float=True)
        # this run saline sim eventually calls snap, which adjusts cell boundaries
        return proposals
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
            print(f"t={tt:5.2f} → {self.equilibrium_bar(tt)}")

    def run_dynamics(self, steps=20, total_time=2*pi):
        self.reset_state()
        dt = total_time/steps
        for _ in range(steps+1):
            bar = self.step(dt)
            print(f"t={self.current_t:5.2f} → {bar}")

    def balance_system(self, cells, bitbuffer,
                    mode='open',
                    C_ext=1500.0, p_ext=10000.0,
                    R=8.314, T=298.15,
                    dt=0.1, max_steps=50,
                    sigma=1.0, Ps=1.0):
        """
        Balance container via iterative Kedem–Katchalsky with additional
        feedback terms.  Each cell tracks its own volume ``V_i`` and solute
        quantity ``S_i``.  Concentration ``C_i`` is updated as ``S_i / V_i`` and
        the internal pressure uses a simple turgor model::

            P_i = P0 + k_i (V_i/V0_i - 1)

        Water and solute fluxes are integrated over ``dt`` until volumes stop
        changing or ``max_steps`` is reached.  The external bath accumulates the
        displaced volume and solute, enforcing mass conservation.
        """
        if not cells or mode != 'open':
            return [CellProposal(c) for c in cells]

        # Treat the external bath as a finite reservoir
        bath_volume = sum(c.volume for c in cells)
        bath_solute = C_ext * bath_volume

        proposals = [CellProposal(c) for c in cells]
        for _ in range(max_steps):
            bath_conc = bath_solute / bath_volume if bath_volume > 0 else 0.0
            max_delta = 0.0
            total_delta = 0.0
            for cell in cells:
                # Permeability and elastic parameters are cell specific
                Lp = (cell.l_solvent_permiability + cell.r_solvent_permiability) / 2.0
                Ps_cell = Ps if Ps is not None else Lp
                k = getattr(cell, 'elastic_coeff', 0.0)
                V0 = getattr(cell, 'reference_volume', cell.volume)

                # Update pressure and concentration based on current state
                cell.concentration = cell.salinity / cell.volume
                cell.pressure = cell.base_pressure + k * (cell.volume / V0 - 1.0)

                dP = p_ext - cell.pressure
                dC = bath_conc - cell.concentration

                Jv = Lp * (dP - sigma * R * T * dC)
                deltaV = Jv * dt

                Cavg = 0.5 * (bath_conc + cell.concentration)
                Js = Ps_cell * (dC - (1 - sigma) * Cavg * dP / (R * T))
                deltaS = Js * dt

                cell.volume += deltaV
                cell.salinity += deltaS
                cell.concentration = cell.salinity / cell.volume
                cell.pressure = cell.base_pressure + k * (cell.volume / V0 - 1.0)

                bath_volume -= deltaV
                bath_solute -= deltaS

                max_delta = max(max_delta, abs(deltaV))
                total_delta += deltaV
            if max_delta < 1e-6:
                break

        # Expand bitbuffer if total volume increased
        if total_delta > 0:
            system_lcm = self.lcm(cells)
            delta_bits = bitbuffer.intceil(bitbuffer.round(total_delta), system_lcm)
            if delta_bits > 0:
                manual_event = (None, bitbuffer.mask_size, delta_bits)
                proposals = bitbuffer.expand([manual_event], cells, proposals)

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
    # Example parameters
    n = 10
    t = symbols('t')
    s_exprs = [1 + sin(t + 2*pi*i/n) for i in range(n)]
    p_exprs = [1 + sin(t + pi/4 + 2*pi*i/n) for i in range(n)]

    # Instantiate with bump_under_one and protect flags
    sys_int = SalineHydraulicSystem(
        s_exprs, p_exprs,
        width=97, chars=[chr(97+i) for i in range(n)],
        tau=5, math_type='int',
        int_method='adams',
        protect_under_one=True,
        bump_under_one=True
    )

    # Continuous demo for 50 steps
    sys_int.reset_state()
    dt = (2 * pi) / 40
    for _ in range(4000):
        bar = sys_int.step(dt)
        print(f'\r{bar}', end='\n', flush=True)
        time.sleep(0.1)

    print() 