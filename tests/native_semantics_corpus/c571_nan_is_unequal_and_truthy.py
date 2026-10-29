def root(a: float) -> float:
    # A NaN is unequal to itself and is true, as Python has it.
    big = a * 1e308 * 10.0
    n = big - big
    r = 0.0
    if n != n:
        r += 1.0
    if n:
        r += 2.0
    if not (n == n):
        r += 4.0
    return r
# INPUTS [{'a':1.0},{'a':0.0},{'a':-3.0}]
