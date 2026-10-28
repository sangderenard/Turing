def root(n: int) -> int:
    g = [[1, 2, 3], [4, 5], [n]]
    i = n % 3
    return len(g[i]) * 100 + len(g[0]) * 10 + len(g)
# INPUTS [{"n": 0}, {"n": 1}, {"n": 5}]
