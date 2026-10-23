def push(xs: list[int], v: int) -> None:
    xs.append(v)


def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def root(n: int) -> int:
    xs0 = [4, 0, 8]
    ys0 = [x0 * n for x0 in xs0]
    xs1 = []
    push(xs1, n)
    push(xs1, 6)
    setfirst(xs1, 8)
    return (sum(ys0) * 10 + len(ys0)) + (len(xs1) * 100 + xs1[0] + xs1[1])
# INPUTS [{'n':0},{'n':2},{'n':-3}]
