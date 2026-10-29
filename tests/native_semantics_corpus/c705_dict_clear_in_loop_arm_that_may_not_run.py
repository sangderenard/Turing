def root(n: int) -> int:
    # A dict's ``clear`` an ``if`` holds inside a loop is a mapping effect of
    # the loop; the loop-state record and the call record agree it is one.
    d = {1: n, 2: 5}
    for k in range(3):
        if k > 9:
            d.clear()
    return len(d) * 10 + d[1]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}]
