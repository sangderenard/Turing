def root(n: int) -> int:
    best = 0
    for k in range(n):
        d = k - 2
        if d != 0 and 12 // d > best:
            best = 12 // d
    return best
# INPUTS [{'n':0},{'n':3},{'n':7}]
