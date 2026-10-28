def pick(n, c):
    if c == 2:
        return n
    return n * 0.5

def root(n: int, c: int):
    return pick(n, c)
# INPUTS [{'n':7,'c':2},{'n':7,'c':3}]
