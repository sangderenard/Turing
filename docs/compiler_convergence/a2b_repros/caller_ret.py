def pick(a: int, b: int, c: int):
    if c > 0:
        return (a, b)
    return None

def root(a: int, b: int, c: int) -> int:
    r = pick(a, b, c)
    if r is None:
        return -7
    return r[0] * 10 + r[1]
