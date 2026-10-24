import math


def root(n: int) -> int:
    # Past 2**31: ceil and round must stay exact in every lane.
    x = n * 1000000000.0 + 0.5
    return math.ceil(x) - round(x) + round(n * 2500000000.5) % 1000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}, {'n': 5}, {'n': 9}]
