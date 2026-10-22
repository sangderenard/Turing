def grow(xs: list[int], k: int) -> None:
    for i in range(k):
        xs.append(i)


def push(xs: list[int], v: int) -> None:
    xs.append(v)


def pairlist(x: int) -> list[int]:
    return [x, x + 1]


def root(n: int) -> int:
    xs0 = [2, n]
    grow(xs0, n % 3 + 1)
    xs1 = [n, 4]
    push(xs1, 3)
    ys = pairlist(n + 4)
    zs = [n, 1]
    push(zs, 9)
    return (len(xs1) * 100 + xs1[-1]) + len(xs0) * 10000 + (len(ys) * 10 + ys[1]) * 1000000 + (len(zs) * 100 + zs[-1]) * 100000000

# INPUTS [{'n': -3}, {'n': 0}, {'n': 4}]
