def maybe(row: int, flag: int):
    return row + 1 if flag > 0 else None

def root(row: int, flag: int) -> int:
    v = maybe(row, flag)
    if v is None:
        return -1
    return v
# INPUTS [{'row':4,'flag':1},{'row':4,'flag':0}]
