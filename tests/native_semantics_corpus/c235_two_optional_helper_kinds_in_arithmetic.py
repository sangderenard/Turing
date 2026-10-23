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
    o1 = maybe_inc(n, 2)
    o2 = maybe_guard(n, 2)
    return (-1 if o1 is None else o1) + (-1 if o2 is None else o2) + n
# INPUTS [{'n':-3},{'n':4},{'n':12}]
