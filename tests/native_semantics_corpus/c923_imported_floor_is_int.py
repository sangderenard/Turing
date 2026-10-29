import math
from math import ceil, floor, trunc


def root(x: float):
    # ``floor``/``ceil``/``trunc`` imported by name are ``math.floor`` & co.:
    # their results are ints, as through ``math.``; ``math.floor(x)``
    # returned alone is a value too.
    return floor(x) * 3 + 1, ceil(x), trunc(x), math.floor(x)

# INPUTS [{'x': 2.7}, {'x': -0.6}, {'x': -1.5}, {'x': 2.5}, {'x': -2.5}, {'x': 3000000000.5}]
