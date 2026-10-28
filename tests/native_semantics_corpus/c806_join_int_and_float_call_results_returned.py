def half(n):
    return n * 0.5

def same(n):
    return n

def root(n: int, c: int):
    if c == 2:
        s = same(n)
    else:
        s = half(n)
    return s
# INPUTS [{'n':7,'c':2},{'n':7,'c':3}]
