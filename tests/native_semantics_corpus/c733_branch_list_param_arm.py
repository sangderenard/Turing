def pick(xs: list[int], n: int) -> int:
    if n > 2: t = xs
    else: t = [n, 1]
    return len(t) * 100 + t[0] + t[len(t) - 1]

def root(n: int) -> int:
    base = [4, 5, 6, n]
    return pick(base, n)
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
