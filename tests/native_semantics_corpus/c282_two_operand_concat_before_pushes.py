def push(xs: list[int], v: int) -> None:
    xs.append(v)


def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def root(n: int) -> int:
    xs0 = [n, 2] + [n * 4]
    xs1 = [1]
    push(xs1, n)
    push(xs1, 6)
    return len(xs1) * 100 + xs1[2] + xs0[2]
# INPUTS [{'n':0},{'n':2},{'n':-3}]
