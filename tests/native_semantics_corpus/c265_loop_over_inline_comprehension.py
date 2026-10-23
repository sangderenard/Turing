def root(n: int) -> int:
    xs = [1, 2, 3]
    t = 0
    for y in [x * n for x in xs]:
        t += y
    u = 0
    for z in [w + n for w in [4, 5]]:
        u = u * 10 + z
    return t * 1000 + u
# INPUTS [{'n':1},{'n':4},{'n':-2}]
