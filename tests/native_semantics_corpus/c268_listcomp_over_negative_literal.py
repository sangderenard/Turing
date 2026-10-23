def root(n: int) -> int:
    ys = [x * n for x in [5, -1]]
    ys.append(n + 1)
    return len(ys) * 100 + sum(ys) + ys[1] * 1000
# INPUTS [{'n':1},{'n':-3},{'n':0}]
