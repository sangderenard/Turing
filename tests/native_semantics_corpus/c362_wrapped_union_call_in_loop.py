def maybe(row: int, flag: int):
    return row + 1 if flag > 0 else None

def wrap(row: int, flag: int):
    return maybe(row, flag)

def root(row: int, flag: int) -> int:
    s = 0
    for i in range(3):
        v = wrap(row + i, flag - i)
        if v is not None:
            s += v
    return s
# INPUTS [{'row':4,'flag':1},{'row':4,'flag':0},{'row':1,'flag':3}]
