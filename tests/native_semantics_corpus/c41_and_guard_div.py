def root(d: int) -> int:
    total = 0
    if d != 0 and 10 // d > 1:
        total = 1
    return total
# INPUTS [{'d':0},{'d':2},{'d':20}]
