# ``np.asarray`` read only as a tensor-namespace callee is a static symbol,
# not a region value: the call carries the typed result.
import numpy as np


def root(n: int) -> float:
    xs = [n * 1000.5, 2.25]
    a = np.asarray(xs)
    return a[0] + a[1]

# INPUTS [{'n': 3}, {'n': 5}, {'n': -7}]
