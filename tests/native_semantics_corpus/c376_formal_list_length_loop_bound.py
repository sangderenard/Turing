def weigh(xs: list[int]) -> int:
    t = 0
    for i in range(len(xs)):
        t += xs[i] * i
    return t


def steps(xs: list[int]) -> int:
    k = len(xs)
    t = 0
    for i in range(1, k):
        t += xs[i] - xs[i - 1]
    return t


def grow(xs: list[int], n: int) -> int:
    xs.append(n)
    t = 0
    for i in range(len(xs)):
        t += xs[i] * (i + 1)
    return t


def root(n: int) -> int:
    return (
        weigh([1, n, 3]) * 10000
        + steps([1, n, 3, n * n]) * 100
        + grow([2, n], n)
    )

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
