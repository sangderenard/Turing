# Lane E (E24): a keyed delete scans with an int64 row counter; a later
# store of a large int row must survive the delete.
def root(n: int) -> int:
    d = {1: n, 2: 3, 5: 7}
    del d[2]
    d[8] = n + 36028797018963969
    return len(d) * 100 + d[8] % 1000 + d[1] + d[5]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
