def grow(xs: list[int], k: int) -> int:
    xs.append(k)
    return len(xs)
def root(n: int) -> int:
    a = [7, 8, 9]
    if n > 2:
        x = a
    else:
        x = [4, 5, 6]
    m = grow(x, n)
    x[0] = 100 + n
    return m * 100000 + a[0] * 10 + a[1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
