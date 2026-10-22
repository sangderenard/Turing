def root(n: int) -> int:
    d = {1: 10, 2: 20, 0: 30}
    return d[1] * 10000 + d[0] * 100 + d[2] + n

# INPUTS [{'n': 0}]
