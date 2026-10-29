def head_sum(cells, count):
    total = cells[0] - cells[0]
    i = 0
    while i < count:
        total += cells[i]
        i += 1
    return total


def ints(xs: list[int], k: int) -> int:
    return head_sum(xs, k)


def floats(ys: list[float], k: int) -> float:
    return head_sum(ys, k)


def root(n: int, f: float) -> float:
    return ints([n, 2, -5, 7], 3) + floats([f, 0.5, 2.5], 3)

# INPUTS [{'n': 4, 'f': 1.5}, {'n': -3, 'f': 0.25}]
