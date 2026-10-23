def push(xs: list[int], v: int) -> None:
    xs.append(v)


def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def root(n: int) -> int:
    ys0 = [x0 * n for x0 in [4, 0, 8]]
    xs1 = [3]
    push(xs1, n)
    return sum(ys0) * 10 + len(xs1) * 100 + xs1[1]
# INPUTS [{'n':0},{'n':2},{'n':-3}]
