# A dict keyed by tuples and filled in a loop (bootstrap: the identity
# book's rows are tuples).  The key width was read off the store rows at the
# value's position, so the table was born with one key column and its store
# helper refused the two-column key.
def root(n: int) -> int:
    cells = {}
    for i in range(n):
        cells[(i % 2, i)] = i * 7
    return len(cells) * 100 + n
# INPUTS [{'n': 0}, {'n': 1}, {'n': 4}]
