def root(n: int) -> int:
    # A dict ``pop`` statement in the body of a loop that unrolls.
    d = {1: 5, 2: 6}
    for i in range(1):
        d[3] = n + i
        d.pop(2)
    return len(d) * 100 + d.get(2, -1) * 10 + d[3]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
