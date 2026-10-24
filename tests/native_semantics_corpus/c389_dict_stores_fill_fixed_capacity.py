def root(n: int) -> int:
    # Every keyed store checks the table's status, so a store that finds a
    # FIXED table full raises instead of dropping its row.  These stores
    # fill the table exactly: updates reuse rows, a re-stored deleted key
    # takes a fresh one, and a store in an arm counts as a row.
    d = {1: n, 2: 5}
    d[1] = n + 1
    d[3] = 7
    del d[2]
    d[2] = 9
    if n > 0:
        d[4] = n * 3
    d[3] = d[3] + 1
    return len(d) * 1000 + d[1] * 100 + d[2] * 10 + d[3] + d.get(4, 0)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 5}]
