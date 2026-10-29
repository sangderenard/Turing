import math


# Scalar math.log near 1, where the result has no leading digits to spare
# (1+2**-52, 1-2**-53, 1+2**-30, 1-2**-40), at the smallest subnormal and
# normal, and at the largest double.  The signal-core route
# (ir_identities.lower_transcendentals_to_signal_cores) must give the
# correctly rounded double on every row; CPython agrees with mpmath on all.
# (``1.0 *`` is exact; a bare ``return math.log(x)`` is refused before any
# math runs: "a Ret in block entry returns none", a separate defect.)
def root(x: float) -> float:
    return 1.0 * math.log(x)
# INPUTS [{'x': 1.0000000000000002}, {'x': 0.9999999999999999}, {'x': 1.0000000009313226}, {'x': 0.9999999999990905}, {'x': 5e-324}, {'x': 2.2250738585072014e-308}, {'x': 1.7976931348623157e+308}, {'x': 0.1}, {'x': 1.5}]
