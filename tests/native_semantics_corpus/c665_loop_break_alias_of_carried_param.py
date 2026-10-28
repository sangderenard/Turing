def f(v: int) -> int:
    r = 0
    while v < 100:
        if v > 3:
            r = v
            break
        v = v + 1
    return r


def root(n: int) -> int:
    return f(n) + 1
# INPUTS [{'n': 0}, {'n': 7}, {'n': -4}]
