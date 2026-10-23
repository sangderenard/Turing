def root(n: int) -> int:
    xs1 = [3, n]
    xs1.append(2)
    a = xs1[1]
    xs1[1] = 9
    return a + 0
# INPUTS [{"n":0},{"n":2},{"n":-3}]
