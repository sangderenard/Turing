def fork_of(scope: int) -> tuple[int, int] | None:
    if scope > 3:
        return None
    return (scope + 1, scope * 2)

def root(row: int, flag: int) -> int:
    if flag > 0:
        fork = fork_of(row)
    else:
        fork = None
    if fork is None or row < 0:
        return -1
    origin, stamp = fork
    return origin + stamp
# INPUTS [{'row':1,'flag':1},{'row':5,'flag':1},{'row':1,'flag':0},{'row':-2,'flag':1}]
