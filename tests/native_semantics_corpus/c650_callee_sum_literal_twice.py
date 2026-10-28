def total(xs: list[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t


def root(n: int) -> int:
    a = total([1, 2, 3, 4])
    b = total([n, n + 1])
    return a * 100 + b * 10 + n

# INPUTS [{'n': 1}, {'n': 5}, {'n': -2}]
