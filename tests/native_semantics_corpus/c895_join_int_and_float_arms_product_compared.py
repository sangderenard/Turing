def root(n: int, c: int):
    if c == 2:
        s = n
    else:
        s = n * 0.5
    t = s * 3
    return t > 10
# INPUTS [{'n':7,'c':2},{'n':7,'c':3},{'n':2,'c':3}]
