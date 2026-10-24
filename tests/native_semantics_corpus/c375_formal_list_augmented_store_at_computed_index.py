def bump(xs: list[int], n: int) -> None:
    for i in range(4):
        xs[i % 2] += n


def poke(xs: list[int], n: int) -> int:
    xs[n % 2] += n
    return xs[0] + xs[1]


def root(n: int) -> int:
    xs = [2**55 + 1, 3]
    bump(xs, n)
    ys = [1, 3]
    t = poke(ys, n)
    return (xs[0] - 2**55) * 1000 + xs[1] + ys[0] * 7 + ys[1] * 5 + t

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
