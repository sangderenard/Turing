def root(n):
    g = [[1.5, 2.5], [n * 0.5, 4.0, 1.0]]
    s = 0.0
    for r in g:
        for x in r:
            s = s + x
    for x in g[n % 2]:
        s = s * 2.0 + x
    return s
# INPUTS [{"n": 0}, {"n": 3}]
