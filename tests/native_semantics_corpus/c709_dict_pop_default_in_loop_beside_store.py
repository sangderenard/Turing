def root(n: int) -> int:
    # A ``pop`` with a default in a loop body that also stores a new key:
    # the loop carries the dict, and the pop is one of its keyed effects.
    d = {1: 10, 2: 20}
    for i in range(n):
        d.pop(1, None)
        d[3] = i
    return len(d) * 100 + d.get(1, -1) * 10 + d.get(3, -2)

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
