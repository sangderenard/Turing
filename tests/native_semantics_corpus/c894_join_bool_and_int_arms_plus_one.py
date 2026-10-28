def root(n: int, c: int):
    if c == 2:
        s = n > 3
    else:
        s = n
    return s + 1
# INPUTS [{'n':7,'c':2},{'n':7,'c':3}]
