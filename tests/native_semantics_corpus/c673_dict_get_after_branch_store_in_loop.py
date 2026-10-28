def root(n: int) -> int:
    # Two ``get`` reads of a dict a loop's ``if`` stores into: the loop's
    # carried value is the branch merge, and the reads are int rows.
    d = {1: n, 7: 4}
    for i in range(4):
        if i == n:
            d[1] = i
    return d.get(7, -1) + d.get(1, 0)

# INPUTS [{'n': -3}, {'n': 2}]
