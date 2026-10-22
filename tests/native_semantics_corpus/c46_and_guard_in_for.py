def root(n: int) -> int:
    total = 0
    for k in range(n):
        d = k - 2
        if d != 0 and 12 // d > 1:
            total += 12 // d
    return total
# INPUTS [{'n':0},{'n':3},{'n':7}]
