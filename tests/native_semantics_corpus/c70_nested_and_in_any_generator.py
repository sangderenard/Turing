def root(n: int) -> int:
    return 1 if any(x > 0 and (x < n and (x != 3 and x * 2 + 1 > n)) for x in range(6)) else 0

# INPUTS [{'n': 0}, {'n': 4}, {'n': 6}, {'n': 9}]
