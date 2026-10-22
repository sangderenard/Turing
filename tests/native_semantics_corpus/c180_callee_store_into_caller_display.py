def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def root(n: int) -> int:
    xs = [n, 4]
    setfirst(xs, 9)
    return len(xs) * 100 + xs[0]

# INPUTS [{'n': 1}, {'n': 5}]
