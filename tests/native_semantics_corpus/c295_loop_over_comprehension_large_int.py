def root(n: int) -> int:
    xs = [1, 2, 3]
    t = 0
    for y in [x * n for x in xs]:
        t += y
    return t
# INPUTS [{'n':4},{'n':36028797018963969},{'n':-36028797018963971}]
