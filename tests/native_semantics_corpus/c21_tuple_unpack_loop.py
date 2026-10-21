def root(n: int) -> int:
    pairs = ((1, 2), (3, 4), (5, 6))
    s = 0
    for a, b in pairs:
        s += a * b + n
    return s
# INPUTS [{'n':0},{'n':2}]
