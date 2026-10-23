def push(xs: list[int], v: int) -> None:
    xs.append(v)


def setfirst(xs: list[int], v: int) -> None:
    xs[0] = v


def root(n: int) -> int:
    t = 0
    for y in [x * n for x in [3, 5, 0]]:
        t += y
    xs = []
    push(xs, n)
    push(xs, 1)
    ys = []
    push(ys, n)
    push(ys, 4)
    setfirst(ys, 2)
    return t + len(xs) * 100 + xs[0] + xs[1] + (len(ys) * 100 + ys[0] * 10 + ys[1]) * 10000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 4}]
