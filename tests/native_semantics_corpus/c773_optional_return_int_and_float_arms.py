def pick(n, c):
    if c == 1:
        return None
    if c == 2:
        return n
    return n * 0.5

def root(n: int, c: int) -> float:
    s = pick(n, c)
    t = 0.0
    if s is not None:
        t = s + 0.25
    return t
# INPUTS [{'n':7,'c':1},{'n':7,'c':2},{'n':7,'c':3}]
