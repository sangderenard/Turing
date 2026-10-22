def root(n: int) -> int:
    d = n
    steps = 0
    while d != 0 and 12 // d > 1:
        d -= 1
        steps += 1
    return steps * 100 + d
# INPUTS [{'n':0},{'n':3},{'n':5}]
