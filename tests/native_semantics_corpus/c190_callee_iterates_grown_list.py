def total(xs: list[int]) -> int:
    s = 0
    for x in xs:
        s += x
    return s


def root(n: int) -> int:
    xs = [n, 4, 7]
    xs.append(n)
    return total(xs) * 10 + len(xs)

# INPUTS [{'n': 1}, {'n': 5}]
