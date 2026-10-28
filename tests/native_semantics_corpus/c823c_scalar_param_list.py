def f(a: list[int], n: int):
    return n * 2, a

def root(n: int) -> int:
    a = [n, 2]
    k, b = f(a, n)
    b.append(5)
    return len(a) * 100 + k * 10 + a[-1] + len(b)
# INPUTS [{'n': 1}, {'n': 3}]
