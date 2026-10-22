def total(xs: list[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t


def root(n: int) -> int:
    xs = [n, 4, 2]
    return total(xs)

# INPUTS [{'n': 1}, {'n': 5}]
