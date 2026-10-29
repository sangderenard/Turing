import math


# Scalar math.exp at arguments where correct rounding is hard: a classic
# hard case (534.707910443053), results that underflow into subnormals
# (-740.5, -745.0), the last finite binade (709.78), and 192.9234597167931,
# where CPython (glibc) is 1 ulp from the correctly rounded value
# (mpmath: 6.103711313905535e+83).  The signal-core route
# (ir_identities.lower_transcendentals_to_signal_cores) must give the
# correctly rounded double on every row.  (``1.0 *`` is exact; a bare
# ``return math.exp(x)`` is refused before any math runs: "a Ret in block
# entry returns none", a separate defect.)
def root(x: float) -> float:
    return 1.0 * math.exp(x)
# INPUTS [{'x': 534.707910443053}, {'x': -740.5}, {'x': -745.0}, {'x': 709.78}, {'x': 0.000123}, {'x': -0.5}, {'x': 7.5e-17}, {'x': 192.9234597167931}]
