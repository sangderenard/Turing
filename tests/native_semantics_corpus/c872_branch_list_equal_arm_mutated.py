def grow(xs: list[int], k: int) -> int:
    xs.append(k)
    xs[0] = xs[0] + 1
    return len(xs)
def root(n: int) -> int:
    a = [7, 8, 9]
    if n > 2:
        x = a
    else:
        x = [4, 5, 6]
    y = [1, 2, 3] if n > 4 else [n, n, n]
    m = grow(x, n)
    y.append(5)
    a[1] = a[1] + n
    return m * 100000 + a[0] * 10 + a[1] + x[0] + len(y) * 1000 + y[0]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
