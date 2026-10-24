def root(n: int) -> int:
    # A store after a ``del`` lands: of a new key (the region carrying the
    # delete must still run its post-operation store), and of the deleted
    # key itself (which takes a fresh resident row).
    d = {1: 5, 2: n}
    del d[1]
    d[3] = 4
    e = {1: 5, 2: n}
    del e[1]
    e[1] = 7
    return len(d) * 1000 + d[3] * 100 + len(e) * 10 + e[1] + int(1 in e) * 7

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
