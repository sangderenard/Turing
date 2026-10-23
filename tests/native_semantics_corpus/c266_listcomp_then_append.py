def root(n: int) -> int:
    xs = [1, 2, 3]
    ys = [x * n for x in xs]
    ys.append(n)
    return len(ys) * 100 + sum(ys) + ys[3] * 10000
# INPUTS [{'n':1},{'n':4},{'n':-2}]
