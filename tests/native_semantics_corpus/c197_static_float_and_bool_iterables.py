def total(xs: list[float]) -> float:
    t = 0.0
    for x in xs:
        t += x * 2.0
    return t


def count_true(bs: list[bool]) -> int:
    c = 0
    for b in bs:
        if b:
            c += 1
    return c


def root(n: int) -> float:
    return total([1.5, 2.0, 0.25]) + count_true([True, False, True]) * 10.0 + n

# INPUTS [{'n': 1}, {'n': 5}]
