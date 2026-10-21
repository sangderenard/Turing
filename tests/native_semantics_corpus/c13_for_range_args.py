def root(n: int, step: int) -> int:
    s = 0
    for i in range(1, n, step):
        s += i * i
    return s
# INPUTS [{'n':10,'step':2},{'n':5,'step':1},{'n':1,'step':3}]
