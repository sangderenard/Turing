def root(n: int) -> int:
    xs0 = [1, n, 1, 4]
    xs0.sort()
    c1 = int(any(i1 * 6 > n for i1 in range(4)))
    return xs0[0] * 100 + c1
# INPUTS [{'n': -2}, {'n': 3}, {'n': 30}]
