def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def root(n: int) -> int:
    xs = [3, n]
    xs.append(1)
    a = xs[0]
    setfirst(xs, 8)
    return a * 10 + xs[0]
# INPUTS [{"n":0},{"n":2},{"n":-3}]
