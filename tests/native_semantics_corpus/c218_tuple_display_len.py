def root(n: int) -> int:
    t = (n > 0, n > 2, n > 4)
    u = (n + 1, n * 3)
    a, b = u
    s = 0
    for x in u:
        s += x
    if t[1]:
        return len(t) * 1000 + len(u) * 100 + (a - b) * 10 + s
    return len(t) * 100 + len(u) * 10 + s
# INPUTS [{'n':-1},{'n':1},{'n':3},{'n':5}]
