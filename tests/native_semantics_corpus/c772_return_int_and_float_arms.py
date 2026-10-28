def pick(n, c):
    if c == 2:
        return n
    return n * 0.5

def root(n: int, c: int) -> float:
    s = pick(n, c)
    return s + 0.25
# INPUTS [{'n':7,'c':2},{'n':7,'c':3}]
