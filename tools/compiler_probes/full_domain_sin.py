"""Full-domain sine: a wide reduction feeding a proven core.

The core is exact on its own octant and useless outside it, so the whole
question is the reduction. Doing it in double throws away the precision
before the core ever runs -- an argument of a trillion arrives wrong in
the fourth decimal -- so the subtraction is done in limbs against a tau
derived to the same width.

The integer quadrant is deliberately NOT wide. k is an integer, exactly
representable in a double up to 2**53, and computing it from the
collapsed argument is exact wherever the argument is; what has to be wide
is the SUBTRACTION of k quarter-turns, because that is where the digits
cancel.
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mpmath
import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.extended_precision import Precision
from src.common.tensors.signal_symbolic import (
    CORE_RADII, constant_limbs, evaluate_proof,
)

mpmath.mp.dps = 80


def exact(value: Precision):
    rows = value.to_float_lists()
    return [
        sum((Fraction(row[index]) for row in rows), Fraction())
        for index in range(len(rows[0]))
    ]


def sin_full(values, width: int):
    """sin over the whole real line, reduced wide, evaluated on the core."""

    wide = Precision.of(
        AbstractTensor.get_tensor(np.asarray(values, dtype=np.float64)), width
    )
    quarter = Precision.constant(
        wide, constant_limbs("tau", width, scale=Fraction(1, 4))
    )

    # The quadrant index: an integer, so a double holds it exactly.
    collapsed = np.asarray(wide.collapse().tolist(), dtype=np.float64)
    quarter_double = float(sum(
        Fraction(part) for part in constant_limbs("tau", 1, scale=Fraction(1, 4))
    ))
    index = np.floor(collapsed / quarter_double + 0.5)

    # The subtraction that has to be wide.
    steps = Precision.of(
        AbstractTensor.get_tensor(index), width
    )
    residual = wide - steps * quarter

    # Which core, and which sign, follow from the quadrant.
    quadrant = np.mod(index, 4.0)
    sine = exact(evaluate_proof("sin", residual, width))
    cosine = exact(evaluate_proof("cos", residual, width))
    answer = []
    for position, which in enumerate(quadrant):
        if which == 0:
            answer.append(sine[position])
        elif which == 1:
            answer.append(cosine[position])
        elif which == 2:
            answer.append(-sine[position])
        else:
            answer.append(-cosine[position])
    return answer, exact(residual)


POINTS = [0.3, 3.0, 100.0, 1.0e6, 1.0e12]

print(f"{'x':>10}  {'|r|<=pi/4':>10}  {'width 2':>12}  {'width 4':>12}  {'libm':>12}")
for width in (2, 4):
    pass
results = {}
for width in (2, 4):
    answers, residuals = sin_full(POINTS, width)
    results[width] = answers
    if width == 2:
        within = all(
            abs(float(value)) <= CORE_RADII["sin"] + 1e-9 for value in residuals
        )

for position, point in enumerate(POINTS):
    truth = mpmath.sin(mpmath.mpf(point))
    def error(value):
        return float(abs(
            mpmath.mpf(value.numerator) / value.denominator - truth
        ))
    libm = float(abs(mpmath.mpf(float(np.sin(point))) - truth))
    print(
        f"{point:10g}  {str(within):>10}  "
        f"{error(results[2][position]):12.2e}  "
        f"{error(results[4][position]):12.2e}  {libm:12.2e}"
    )
