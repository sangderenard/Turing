def root(d: int) -> int:
    if d == 0 or 10 // d > 3:
        return 1
    return 2
# INPUTS [{'d':0},{'d':2},{'d':5}]
