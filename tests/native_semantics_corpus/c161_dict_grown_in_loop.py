def root(n: int) -> int:
    d0 = {}
    for i0 in range(n % 4 + 1):
        d0[i0 % 2] = i0
    return len(d0) * 10 + d0[0]
# INPUTS [{'n': 0}, {'n': 3}, {'n': 6}]
