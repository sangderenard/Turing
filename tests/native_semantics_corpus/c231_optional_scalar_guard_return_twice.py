def maybe(row: int, flag: int):
    if flag > 0:
        return row + 1
    return None

def root(row: int, flag: int) -> int:
    v = maybe(row, flag)
    w = maybe(row * 3, flag - 1)
    total = 0
    if v is not None:
        total += v
    if w is None:
        total += 100
    return total
# INPUTS [{'row':4,'flag':2},{'row':4,'flag':1},{'row':4,'flag':0}]
