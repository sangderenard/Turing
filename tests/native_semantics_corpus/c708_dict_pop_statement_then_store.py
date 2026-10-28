def root(n: int) -> int:
    # A dict ``pop`` statement is a keyed delete: it was an empty region, so
    # the popped key stayed in the dict.
    d = {1: 10, 2: 20}
    d.pop(1, None)
    d[3] = n
    return len(d) * 100 + d.get(1, -1) * 10 + d[3]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
