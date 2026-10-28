def pick(xs: list[int], n: int) -> int:
    if n > 2:
        t = xs
    else:
        t = [n, 1]
    t.append(n + 7)
    return len(t) * 100 + t[0] + t[len(t) - 1]

def root(n: int) -> int:
    base = [4, 5, 6, n]
    r = pick(base, n)
    return r * 10 + len(base)
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
