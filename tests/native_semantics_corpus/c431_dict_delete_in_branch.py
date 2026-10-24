def root(n: int) -> int:
    # A ``del`` in an arm with no other statement runs exactly when the arm
    # is taken; ``len`` and ``get`` read the table after it.
    d = {1: n, 2: 5}
    if n > 0:
        del d[2]
    else:
        del d[1]
    return len(d) * 100 + d.get(2, 0) * 10 + d.get(1, 0)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
