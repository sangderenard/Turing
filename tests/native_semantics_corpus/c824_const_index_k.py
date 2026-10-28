def f(xs: list[int], k: int) -> int:
    return xs[k] if k < 2 else 0
def root(n: int) -> int:
    a = [n, 3, 4]
    return f(a, 1) + f(a, 5) + n
# INPUTS [{'n': 1}, {'n': 4}]
