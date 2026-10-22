def root(row: int, flag: int) -> int:
    if flag > 0:
        fork = (row + 1, row * 2)
    else:
        fork = None
    if fork is None:
        return -1
    origin, stamp = fork
    return origin + stamp
# INPUTS [{'row':1,'flag':1},{'row':5,'flag':0}]
