def positives(xs: list[int]) -> int:
    t = 0
    for x in xs:
        if x > 0:
            t += x
    return t


def root(n: int) -> int:
    return positives([4, -2, 7, -9, 1]) * 10 + n

# INPUTS [{'n': 1}, {'n': 5}]
