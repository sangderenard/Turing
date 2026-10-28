def root(n):
    d = {1: [], 2: [n, n]}
    return len(d[1]) * 10 + len(d[2])
# INPUTS [{"n": 0}, {"n": 4}]
