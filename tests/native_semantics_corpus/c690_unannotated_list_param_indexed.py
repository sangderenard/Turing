def span_total(atoms, begin, end):
    index = begin
    total = 0
    while index < end:
        atom = atoms[index]
        total += atom * 3
        index += 1
    return total


def forward(xs: list[int], lo: int) -> int:
    return span_total(xs, lo, len(xs))


def root(n: int) -> int:
    return forward([n, 2, -5, 7], 1)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
