import math


def root(n: int) -> int:
    # ``math.floor``/``ceil``/``trunc`` of a float are ints; of an int,
    # the int itself.
    x = n * 1.25 - 0.5
    a = math.trunc(x)
    b = math.floor(x)
    c = math.ceil(x)
    return a * 10000 + b * 100 + c + math.floor(n) + round(n)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}, {'n': 5}, {'n': 9}]
