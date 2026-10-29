import math


def root(n: int, x: float):
    # ``round``/``math.floor``/``math.ceil`` of a computed float feed int
    # arithmetic (round(2.7) is 3, round(-1.5) is -2, floor(-1.5) is -2).
    y = x * 0.9 + n
    r = round(y)
    return r * 10 + n, math.floor(y) * 100 - math.ceil(y), round(n * 2.5)

# INPUTS [{'n': 0, 'x': 3.0}, {'n': 1, 'x': -1.6666666666666667}, {'n': -1, 'x': 0.5}, {'n': 3, 'x': 2.5}, {'n': -3, 'x': -2.5}, {'n': 5, 'x': 1.5}, {'n': 0, 'x': 0.6666666666666666}]
