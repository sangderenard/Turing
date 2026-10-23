def root(n: int) -> int:
    ys = {x * n for x in [4, 0, 8]}
    ys.add(n + 5)
    return len(ys) * 100 + sum(ys)
# INPUTS [{'n':0},{'n':2},{'n':-3}]
