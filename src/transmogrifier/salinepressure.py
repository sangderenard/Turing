from sympy import symbols, sin, lambdify
from math import pi, ceil, floor

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