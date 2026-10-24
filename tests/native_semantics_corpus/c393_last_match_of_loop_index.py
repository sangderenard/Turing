def root(n: int) -> int:
    # A conditional arm in a loop binds the bare loop index: the merge must
    # version s, not the index the latch increments.
    s = 7
    for i in range(n):
        if i > 1:
            s = i
    return s

# INPUTS [{'n': 0}, {'n': 2}, {'n': 4}, {'n': 6}]
