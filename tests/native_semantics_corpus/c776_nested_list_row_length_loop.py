def root(n: int) -> int:
    g = [[1, 2, 3], [4, 5], [n, 7, 8, 9]]
    total = 0
    for k in range(n):
        total = total + len(g[k % 3]) * (k + 1)
    if len(g[n % 3]) > 2:
        total = total + 1000
    return total
# INPUTS [{"n": 0}, {"n": 1}, {"n": 4}, {"n": 5}]
