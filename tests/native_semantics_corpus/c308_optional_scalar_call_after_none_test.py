def maybe_twice(x: int, y: int):
    return x * 2 if x != y else None


def root(n: int) -> int:
    o = maybe_twice(n, 2)
    t = 5
    if o is not None:
        t = o - 2
    return t * 10 + n

# INPUTS [{'n': 2}, {'n': 7}, {'n': -3}]
