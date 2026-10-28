def root(n: int) -> int:
    # A keyed delete under an ``if`` in a loop, read back with ``get``.
    d = {1: n, 2: 5, 3: 8}
    for i in range(4):
        if i == n:
            del d[2]
    return len(d) * 100 + d.get(2, -1) * 10 + d[3]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 3}]
