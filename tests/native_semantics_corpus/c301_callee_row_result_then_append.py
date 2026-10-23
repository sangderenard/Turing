def first(xs: list[int]) -> int:
    return xs[0] + 1


def root(n: int) -> int:
    xs = [n, 3]
    a = first(xs)
    xs.append(a)
    return a * 100 + len(xs) * 10 + xs[2]

# INPUTS [{'n': 1}, {'n': 5}, {'n': -3}]
