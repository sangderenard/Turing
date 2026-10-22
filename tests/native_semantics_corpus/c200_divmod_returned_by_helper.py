def split(total: int, width: int) -> tuple[int, int]:
    return divmod(total, width)


def root(n: int) -> int:
    rows, rest = split(n * 7 + 3, 4)
    xs = [n, 6]
    xs.extend([-2, 0])
    return rows * 100 + rest * 10 + len(xs) + xs[3]

# INPUTS [{'n': 1}, {'n': 5}, {'n': -3}]
