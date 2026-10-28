def root(n: int) -> int:
    g = [[1, 2, 3], [4, 5], [n, 7]]
    total = 0
    for x in g[n % 3]:
        total = total + x
    return total
# INPUTS [{"n": 0}, {"n": 1}, {"n": 5}]
