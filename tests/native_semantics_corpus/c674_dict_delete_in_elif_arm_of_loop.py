def root(n: int) -> int:
    # A keyed delete in an ``elif`` arm of an ``if`` in a loop, read back
    # after the loop: the delete is an authored control effect that runs in
    # its arm, not in a region after the loop.
    d = {1: n, 2: 5, 3: 8}
    for i in range(4):
        if i == n:
            d[3] = i
        elif i == 1:
            del d[2]
    return len(d) * 100 + d.get(2, -1) * 10 + d[3]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 3}]
