def root(n: int, c: int) -> float:
    if c == 2:
        s = n
    else:
        s = n / 2
    return s + 0.25
# INPUTS [{'n':7,'c':2},{'n':7,'c':3}]
