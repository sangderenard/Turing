def root(n: int) -> int:
    big = abs(n - 2**60 - 1)
    return big + abs(n) - 2**60

# INPUTS [{'n': 3}, {'n': -7}]
