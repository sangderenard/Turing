def maybe(row: int, flag: int):
    return row + 1 if flag > 0 else None

def wrap(row: int, flag: int):
    return maybe(row, flag)

def wrap2(row: int, flag: int):
    return wrap(row + 1, flag)

def root(row: int, flag: int) -> int:
    v = wrap2(row, flag)
    t = 3
    if v is not None:
        t = v * 2
    return t
# INPUTS [{'row':4,'flag':1},{'row':4,'flag':0}]
