def width(xs: list[int]) -> int:
    return len(xs) * 10 + xs[0]


def root(n: int) -> int:
    if n > 2:
        t = [n, 1]
    else:
        t = [7, n]
    return width(t) + n

# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
