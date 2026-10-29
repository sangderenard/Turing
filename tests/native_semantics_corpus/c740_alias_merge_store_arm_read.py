def root(n: int) -> int:
    a = [n, 1, 2]
    t = a if n > 2 else [5, 6]
    t[1] = n + 20
    return a[1] * 10 + len(t) + t[1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
