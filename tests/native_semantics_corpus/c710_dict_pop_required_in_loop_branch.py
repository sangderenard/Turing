def root(n: int) -> int:
    # A ``pop`` under an ``if`` in a loop that also stores: the pop runs on
    # the first iteration only.
    d = {1: 5, 2: 6}
    for i in range(n + 2):
        if i == 0:
            d.pop(2)
        d[3] = n + i
    return len(d) * 100 + d.get(2, -1) * 10 + d[3]

# INPUTS [{'n': -1}, {'n': 0}, {'n': 1}, {'n': 2}]
