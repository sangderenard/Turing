def root(n: int) -> int:
    # A length, a get and a membership test read before a later keyed
    # store see the table as it stood, even though the graph numbers the
    # reads after the store.
    d = {1: n, 2: 4}
    t = len(d)
    g = d.get(7, -1)
    m = int(7 in d)
    d[7] = 5
    return t * 10000 + g * 1000 + m * 100 + len(d) * 10 + d[7] + d[1]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 4}, {'n': 9}]
