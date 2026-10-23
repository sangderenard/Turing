def helper(x: int, y: int) -> int:
    if x > y:
        return x - y
    return y * 2


def build(m: int) -> list[int]:
    out = []
    for i in range(m):
        out.append(i * i)
    return out


def pair(x: int, y: int) -> tuple[int, int]:
    return x // y, x % y


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


def first_over(x: int, y: int) -> int:
    for i in range(10):
        if i * y > x:
            return i
    return -1


def root(n: int) -> int:
    xs0 = [6, 8, 6, n]
    o1 = maybe_inc(n, 1)
    return (xs0.count(6) * 10 + xs0.index(8)) + ((-1 if o1 is None else o1))
# INPUTS [{'n':-3},{'n':0},{'n':5},{'n':12}]
