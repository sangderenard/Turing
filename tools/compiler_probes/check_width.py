"""Does --limbs actually change the arithmetic? Measure, do not assume."""
import sys
from fractions import Fraction
import numpy as np, mpmath
sys.path.insert(0, "C:/dev/Powershell/turing")
from tools.demo_kuramoto_field import (
    kuramoto_equation, materialise, core_terms, NEIGHBOURS,
)
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.extended_precision import Precision
from src.common.tensors.signal_symbolic import constant_rational, limb_decomposition

mpmath.mp.dps = 80
# One cell, one neighbour pattern, arguments deliberately far from zero so
# the fold is doing real work.
theta0 = [0.3, 2.9, -17.25, 1e3]
nbrs = {"up": [1.1, -4.0, 3.5, -1e3], "down": [0.0, 6.28, -1.0, 7.0],
        "left": [-2.2, 1.0, 0.5, 2.0], "right": [3.3, -1.5, 9.0, -0.25]}
K, DT = Fraction(4, 5), Fraction(1, 20)

def truth():
    out = []
    for i in range(len(theta0)):
        t = mpmath.mpf(theta0[i])
        pull = sum(mpmath.sin(mpmath.mpf(nbrs[n][i]) - t) for n in NEIGHBOURS)
        out.append(t + mpmath.mpf(DT.numerator)/DT.denominator *
                   (0 + mpmath.mpf(K.numerator)/K.denominator * pull))
    return out

exact = truth()
print(f"{'limbs':>6} {'digits':>7} {'terms':>6} {'worst error':>14}")
for limbs in (1, 2, 3, 4):
    digits = max(17, 16 * limbs)
    sine = list(core_terms("sin", digits)); cosine = list(core_terms("cos", digits))
    terms = max(len(sine), len(cosine))
    sine += [Fraction(0)] * (terms - len(sine))
    cosine += [Fraction(0)] * (terms - len(cosine))
    eq, consts = kuramoto_equation(terms)
    step, params, _src = materialise(eq, "kuramoto_step")
    theta = Precision.of(AbstractTensor.get_tensor(np.array(theta0)), limbs)
    def C(v):
        return Precision.constant(theta, tuple(float(p) for p in limb_decomposition(v, limbs)))
    q = constant_rational("tau", digits) / 4
    supply = {"omega": C(Fraction(0)), "coupling": C(K), "dt": C(DT),
              "quarter": C(q), "inv_quarter": C(1/q), "theta": theta}
    for n in NEIGHBOURS:
        supply[n] = Precision.of(AbstractTensor.get_tensor(np.array(nbrs[n])), limbs)
    for n_, v_ in consts.items():
        supply[n_] = C(v_)
    for pre, vals in (("c", sine), ("d", cosine)):
        for i, v in enumerate(vals):
            supply[f"{pre}{i}"] = C(v)
    got = step(**{p: supply[p] for p in params})
    limbsum = [sum(mpmath.mpf(float(t.tolist()[i])) for t in got.terms())
               for i in range(len(theta0))]
    worst = max(abs(a - b) for a, b in zip(limbsum, exact))
    print(f"{limbs:>6} {digits:>7} {terms:>6} {float(worst):14.3e}")
