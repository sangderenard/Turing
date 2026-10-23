def maybe(row: int, flag: int):
    if flag > 0:
        r = row + 1
    else:
        r = None
    return r

def root(row: int, flag: int) -> int:
    v = maybe(row, flag)
    if v is None:
        return -1
    return v
# INPUTS [{'row':4,'flag':1},{'row':4,'flag':0}]
