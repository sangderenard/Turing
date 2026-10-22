def pair(scope: int) -> tuple[int, int]:
    return (scope + 1, scope * 2)

def root(row: int, flag: int) -> int:
    if flag > 0:
        t = pair(row)
    else:
        t = (7, 8)
    a, b = t
    return a + b
# INPUTS [{'row':1,'flag':1},{'row':5,'flag':0}]
