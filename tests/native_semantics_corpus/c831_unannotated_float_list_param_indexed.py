def weighted(cells, begin, end):
    index = begin
    total = 0.0
    while index < end:
        cell = cells[index]
        total += cell * 0.5
        index += 1
    return total


def forward(xs: list[float], lo: int) -> float:
    return weighted(xs, lo, len(xs))


def root(n: float) -> float:
    return forward([n, 2.5, -5.0, 7.0], 1)

# INPUTS [{'n': -3.0}, {'n': 0.5}, {'n': 2.0}]
