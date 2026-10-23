def root(n: int) -> int:
    ys = [x + n for x in [1, 2]]
    for i in range(3):
        ys.append(i * n)
    return len(ys) * 100 + sum(ys) + max(ys) * 10000
# INPUTS [{'n':1},{'n':4},{'n':-2}]
