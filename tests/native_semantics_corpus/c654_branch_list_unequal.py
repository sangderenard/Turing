def total(xs: list[int]) -> int:
    t = 0
    for x in xs: t += x
    return t
def root(n: int) -> int:
    if n > 2: t = [n, 1, 2]
    else: t = [7, n]
    return total(t) + len(t) * 100 + n
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
