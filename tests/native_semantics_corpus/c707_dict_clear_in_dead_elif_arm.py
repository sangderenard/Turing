def root(n: int) -> int:
    # A ``clear`` in an ``elif`` arm that never runs must not change the
    # dict: the effect runs under both of its conditions, the ``elif`` test
    # and the failed ``if`` test, not under the outer one alone.
    d = {}
    for k in range(n + 3):
        if k % 2 == 0:
            d[k] = k
        elif k > 5:
            d.clear()
    return len(d)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
