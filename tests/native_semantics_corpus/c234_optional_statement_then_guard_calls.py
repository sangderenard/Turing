def maybe_inc(x: int, y: int):
    if x > y:
        r = x + y
    else:
        r = None
    return r


def maybe_guard(x: int, y: int):
    if x < y:
        return None
    return x - y


def root(n: int) -> int:
    o1 = maybe_inc(n, 3)
    o2 = maybe_guard(n, 5)
    t2 = 0
    if o2 is not None:
        t2 = o2 * 3
    return t2 + (0 if o1 is None else 1)
# INPUTS [{'n':-3},{'n':4},{'n':12}]
