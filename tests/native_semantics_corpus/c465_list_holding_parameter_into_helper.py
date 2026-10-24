def use(rec: list[int], x: int) -> int:
    k = rec[0]
    return x + k


def root(n: int) -> int:
    g = [n, 0]
    return use(g, 3)
# INPUTS [{"n":0},{"n":4}]
