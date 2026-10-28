def root(n):
    d = {1: [1, 2], 2: [3, n]}
    total = 0
    for k in d:
        for x in d[k]:
            total = total + x * k
    return total
# INPUTS [{"n": 0}, {"n": 4}, {"n": -2}]
