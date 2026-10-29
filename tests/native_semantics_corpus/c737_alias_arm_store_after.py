def root(n: int) -> int:
    a = [n, 1]
    if n > 2:
        t = a
    else:
        t = [7, n, 2]
    a[0] = a[0] + 5
    return t[0] + len(t)
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
