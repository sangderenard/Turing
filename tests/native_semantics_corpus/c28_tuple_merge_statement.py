def root(row: int, flag: int) -> int:
    if flag > 0:
        t = (row + 1, row * 2)
    else:
        t = (7, 8)
    a, b = t
    return a + b
# INPUTS [{'row':1,'flag':1},{'row':5,'flag':0}]
