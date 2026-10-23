def root(n: int) -> int:
    ys = {x % 2 + n for x in [1, 3, 5]}
    ys.add(n + 7)
    ys.add(n + 1)
    return len(ys) * 100 + sum(ys)
# INPUTS [{'n':0},{'n':2},{'n':-3}]
