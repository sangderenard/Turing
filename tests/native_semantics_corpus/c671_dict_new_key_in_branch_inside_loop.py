def root(n: int) -> int:
    # A new key stored under an ``if`` in a loop is present after the loop
    # only on the iterations that took the branch.
    d = {1: n}
    for i in range(4):
        if i == n:
            d[i + 5] = i
    return len(d) * 100 + d.get(7, -1) * 10 + d[1]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 3}]
