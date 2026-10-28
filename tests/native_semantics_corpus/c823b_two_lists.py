def mk(n: int):
    xs = [n, 1]
    ys = [2, n]
    return xs, ys

def root(n: int) -> int:
    xs, ys = mk(n)
    xs.append(5)
    ys.append(7)
    return len(xs) * 1000 + len(ys) * 100 + xs[-1] * 10 + ys[-1]
# INPUTS [{'n': 1}, {'n': 3}]
