def root(n: int) -> int:
    q = divmod(n + 7, 5)
    a, b = divmod(n - 9, 4)
    return q[0] * 100 + q[1] * 10 + a * 1000 + b

# INPUTS [{'n': 1}, {'n': 5}, {'n': -3}, {'n': 12}]
