def root(n: int) -> int:
    # A row of two members times a literal count is the display of the
    # repeated members, in order.
    a = [n, 7] * 3
    a[1] = n + 1
    return len(a) * 100 + a[0] + a[5] + a[4]

# INPUTS [{'n': 3}, {'n': 4}]
