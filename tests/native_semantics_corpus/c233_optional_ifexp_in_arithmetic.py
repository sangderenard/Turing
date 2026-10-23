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
    o1 = maybe_guard(n, 3)
    return (-1 if o1 is None else o1) + n
# INPUTS [{'n':-3},{'n':4},{'n':12}]
