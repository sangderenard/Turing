def root(a: float, b: float) -> float:
    # Builtin max/min keep the first argument unless a later one is
    # strictly better, so a NaN wins only when it comes first.
    big = a * 1e308 * 10.0
    n = big - big
    r = 0.0
    if max(n, b) != max(n, b):
        r += 1.0
    if max(b, n) == b:
        r += 2.0
    if min(n, b) != min(n, b):
        r += 4.0
    if min(b, n) == b:
        r += 8.0
    return r + max(a, b) - min(a, b)
# INPUTS [{'a':1.0,'b':3.0},{'a':0.0,'b':3.0},{'a':-2.0,'b':-5.0},{'a':4.0,'b':4.0}]
