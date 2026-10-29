import math


# Scalar math.sin/cos/tan at arguments whose reduction is hard: 1e22 and
# 1e300 (Payne-Hanek territory), the smallest subnormal, the doubles nearest
# pi/2 and pi, and 349.0426838931851, where CPython (glibc) sin is 1 ulp from
# the correctly rounded value.  The signal-core route
# (ir_identities.lower_transcendentals_to_signal_cores -> circular_double)
# must give the correctly rounded double on every row.  (``1.0 *`` is
# exact; a bare ``return math.sin(x)`` is refused before any math runs: "a
# Ret in block entry returns none", a separate defect.)
def root(x: float) -> float:
    return 1.0 * math.sin(x) + 4.0 * math.cos(x) + 16.0 * math.tan(x)
# INPUTS [{'x': 1e22}, {'x': 1e300}, {'x': 5e-324}, {'x': 1.5707963267948966}, {'x': 3.141592653589793}, {'x': 0.5}, {'x': 349.0426838931851}, {'x': -707.3548455089598}]
