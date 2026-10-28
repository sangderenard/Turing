def root(n: int) -> int:
    # A full table closes up its deleted rows to take a new key; the keys
    # still read back their values.
    d = {1: n, 2: 5, 3: 7}
    for i in range(3):
        del d[1]
        d[1] = i + n
        del d[2]
        d[2] = i
    return d[3] * 1000 + d[1] * 100 + d[2] * 10 + len(d)

# INPUTS [{'n': 0}, {'n': 4}]
