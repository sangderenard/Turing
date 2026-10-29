def root(n):
    d = {1: [1, 2], 2: [3, n]}
    return d[1 + n % 2][0] + d[2 - n % 2][1]
# INPUTS [{"n": 0}, {"n": 3}, {"n": 4}]
