def count_true(bs: list[bool]) -> int:
    c = 0
    for b in bs:
        if b:
            c += 1
    return c


def root(n: int) -> int:
    a = count_true([True, True, False, True])
    b = count_true([False, n > 2])
    return a * 100 + b * 10 + n

# INPUTS [{'n': 1}, {'n': 3}]
