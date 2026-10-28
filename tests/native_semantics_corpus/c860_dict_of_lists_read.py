def root(n):
    d = {1: [1, 2, 3], 2: [4, n]}
    return d[2][1] + d[1][2]
# INPUTS [{"n": 0}, {"n": 7}, {"n": -3}]
