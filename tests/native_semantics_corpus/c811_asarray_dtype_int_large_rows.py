# ``np.asarray(xs, dtype=int)`` truncates float rows to 64-bit ints; rows
# beyond 2**31 keep their value.
import numpy as np


def root(n: int) -> int:
    xs = [n * 1000.5, 2.25]
    a = np.asarray(xs, dtype=int)
    return int(a[0]) + int(a[1])

# INPUTS [{'n': 3}, {'n': 5000000}, {'n': -4000000}]
