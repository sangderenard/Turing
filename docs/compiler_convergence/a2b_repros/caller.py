def pick(a: int, b: int, c: int):
    fork = None
    if c > 0:
        fork = (a, b)
    return fork

def root(a: int, b: int, c: int) -> int:
    r = pick(a, b, c)
    if r is None:
        return -7
    return r[0] * 10 + r[1]
