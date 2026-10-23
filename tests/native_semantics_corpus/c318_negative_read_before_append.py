def root(n: int) -> int:
    xs = [3, n]
    a = xs[-1]
    xs.append(7)
    return a * 10 + xs[-1]
# INPUTS [{"n":0},{"n":2},{"n":-3}]
