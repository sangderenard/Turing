def root(n: int) -> int:
    c0 = abs(n - 5)
    ws1 = [2 - n, 5 - n]
    return c0 + sum(ws1)

# INPUTS [{'n': 0}, {'n': 9}]
