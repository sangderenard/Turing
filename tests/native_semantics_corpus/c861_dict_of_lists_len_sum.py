def root(n):
    d = {1: [1, 2, 3], 2: [4, n]}
    return len(d[1]) * 100 + sum(d[2]) + sum(d[1])
# INPUTS [{"n": 0}, {"n": 7}, {"n": -3}]
