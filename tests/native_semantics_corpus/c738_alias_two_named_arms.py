def root(n: int) -> int:
    a = [n, 1]
    b = [4, n, 6]
    if n > 2:
        t = a
    else:
        t = b
    t.append(n + 10)
    a[0] = a[0] + 3
    return a[0] * 1000 + len(a) * 100 + b[0] * 10 + len(b) + t[-1] + t[0]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
