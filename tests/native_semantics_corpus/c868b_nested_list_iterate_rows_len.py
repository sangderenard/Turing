def root(n):
    g = [[1, 2, 3], [4, 5], [n, 7, 8, 9]]
    total = 0
    for r in g:
        total = total * 10 + len(r)
        for x in r:
            if x > n:
                total = total + x
    return total
# INPUTS [{"n": 0}, {"n": 3}, {"n": 8}]
