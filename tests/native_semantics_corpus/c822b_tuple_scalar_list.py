def mk(n: int):
    xs = [n, 1]
    return n + 1, xs

def root(n: int) -> int:
    k, xs = mk(n)
    xs.append(5)
    return len(xs) * 100 + k * 10 + xs[-1]
# INPUTS [{'n': 1}, {'n': 3}]
