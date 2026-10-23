def maybe(row: int, flag: int):
    return row + 1 if flag > 0 else None

def wrap(row: int, flag: int):
    return maybe(row, flag)

def root(row: int, flag: int) -> int:
    v = wrap(row, flag)
    if v is None:
        return -7
    return v
# INPUTS [{'row':4,'flag':1},{'row':4,'flag':0}]
