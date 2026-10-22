def root(n: int) -> int:
    xs = [n, n + 1]
    t = (xs[0], len(xs), n > 2)
    xs.append(7)
    a, b = (n + 1, n * 3)
    if t[2]:
        return len(t) * 1000 + t[1] * 100 + len(xs) * 10 + len((a, b))
    return len(t) * 100 + (a - b) * 10 + len((n, n + 1))
# INPUTS [{'n':1},{'n':3},{'n':5}]
