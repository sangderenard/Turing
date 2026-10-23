def maybe(row: int, flag: int):
    return row + 1 if flag > 0 else None

def root(row: int, flag: int) -> int:
    a = maybe(row, flag)
    b = maybe(row, flag - 1)
    total = 0
    if a is not None:
        total += a
    if b is None:
        total += 100
    return total
# INPUTS [{'row':4,'flag':1},{'row':4,'flag':0},{'row':2,'flag':2}]
