def first(xs: list[int]) -> int:
    return xs[0]


def root(n: int) -> int:
    xs = [n, 4]
    a = first(xs)
    return len(xs) * 100 + a + xs[1]

# INPUTS [{'n': 1}, {'n': 5}]
