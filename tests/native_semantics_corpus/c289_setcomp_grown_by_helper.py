def add1(s: set[int], v: int) -> None:
    s.add(v)


def root(n: int) -> int:
    ys = {x + n for x in [1, 2, 3]}
    add1(ys, 9)
    return len(ys) * 100 + sum(ys)
# INPUTS [{'n':0},{'n':2},{'n':8}]
