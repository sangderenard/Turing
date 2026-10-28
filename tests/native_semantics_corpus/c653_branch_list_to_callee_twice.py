def head2(xs: list[int]) -> int:
    return xs[0] * 10 + xs[1]


def root(n: int) -> int:
    if n > 2:
        t = [n, 1]
    else:
        t = [7, n]
    a = head2(t)
    b = head2(t)
    return a + b + t[0]

# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
