def root(n: int) -> int:
    xs1 = [3, n]
    xs1.append(2)
    a = xs1[1]
    xs1.pop(1)
    return a + len(xs1)
# INPUTS [{"n":0},{"n":2},{"n":-3}]
