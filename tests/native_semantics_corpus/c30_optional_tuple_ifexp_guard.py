def root(row: int, flag: int) -> int:
    fork = (row + 1, row * 2) if flag > 0 else None
    if fork is None or row < 0:
        return -1
    origin, stamp = fork
    return origin + stamp
# INPUTS [{'row':1,'flag':1},{'row':5,'flag':1},{'row':1,'flag':0},{'row':-2,'flag':1}]
