def root(n: int) -> float:
    q, r = divmod(n * 1.5 - 4.0, 2.5)
    return q * 10.0 + r

# INPUTS [{'n': 1}, {'n': 5}, {'n': -3}]
