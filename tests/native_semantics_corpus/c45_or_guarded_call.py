def ratio(d: int) -> int:
    return 30 // d

def root(d: int) -> int:
    if d == 0 or ratio(d) > 4:
        return 1
    return 2
# INPUTS [{'d':0},{'d':2},{'d':10}]
