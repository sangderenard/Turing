def root(n: int) -> int:
    # A keyed store an ``if`` holds inside a loop changes the dict the loop
    # carries: the read after the loop is a table lookup of the carried
    # dict, not a positional read of its initializer's storage.
    d = {1: n, 2: 5}
    for i in range(3):
        if i == n:
            d[2] = i
    return d[2] * 10 + d[1]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
