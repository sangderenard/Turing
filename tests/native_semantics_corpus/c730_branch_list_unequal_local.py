def root(n: int) -> int:
    if n > 2: t = [n, 1, 2]
    else: t = [7, n]
    s = 0
    for v in t: s += v * 2
    return len(t) * 100 + t[0] + t[1] + s
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
