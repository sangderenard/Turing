def size(xs: list[int]) -> int:
    return len(xs) * 10 + xs[0]


def root(n: int) -> int:
    xs = [n, 4]
    xs.append(7)
    return size(xs)

# INPUTS [{'n': 1}, {'n': 5}]
