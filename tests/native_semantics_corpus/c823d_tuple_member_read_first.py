def mk(n: int):
    xs = [n, 1]
    ys = [3]
    return ys, xs

def root(n: int) -> int:
    ys, xs = mk(n)
    xs.append(5)
    ys.append(n)
    return len(xs) * 1000 + len(ys) * 100 + xs[-1] * 10 + ys[-1]
# INPUTS [{'n': 1}, {'n': 3}]
