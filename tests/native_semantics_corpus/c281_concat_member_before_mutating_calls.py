def push(xs: list[int], v: int) -> None:
    xs.append(v)


def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def root(n: int) -> int:
    xs0 = [n] + [3, 6] + [n * 4]
    xs1 = []
    push(xs1, n)
    push(xs1, 6)
    setfirst(xs1, 4)
    return len(xs1) * 100 + xs1[0] + xs1[1] + xs0[3]
# INPUTS [{'n':0},{'n':2},{'n':-3}]
