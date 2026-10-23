def maybe_pick(x: int, y: int):
    return x * 2 if x != y else None

def bump_pick(x: int, y: int):
    s = maybe_pick(x, y)
    return None if s is None else s + 1

def root(n: int) -> int:
    o1 = bump_pick(n, 2)
    return 7 + (-4 if o1 is None else o1)
# INPUTS [{'n':-3},{'n':2},{'n':9}]
