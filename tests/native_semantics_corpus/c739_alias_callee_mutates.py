def grow(xs: list[int], k: int) -> int:
    xs.append(k)
    xs[0] = xs[0] + 1
    return len(xs)
def root(n: int) -> int:
    a = [n, 1]
    if n > 2:
        t = a
    else:
        t = [7, n, 2]
    m = grow(t, n * 2)
    return m * 1000 + len(a) * 100 + a[0] * 10 + a[-1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
