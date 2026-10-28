def root(n: int) -> int:
    if n > 2:
        x = [n, 1, 2]
    else:
        x = [4, 5, n]
    return len(x) * 100 + x[0] * 10 + x[2]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
