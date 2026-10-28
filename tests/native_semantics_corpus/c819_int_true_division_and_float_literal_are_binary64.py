# ``int / int`` is a Python float (binary64), and ``n * 2.5`` takes the float
# literal's own class; neither is a default width.
import math


def root(a: int, b: int) -> float:
    q = a / b
    r = math.sqrt(a) + b * 2.5
    return q + r

# INPUTS [{'a': 7, 'b': 2}, {'a': 1000003, 'b': -9}, {'a': 0, 'b': 5}]
