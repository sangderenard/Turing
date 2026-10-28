def f(a: list[int], n: int):
    return a, n

def root(n: int) -> int:
    a = [n, 2]
    b, k = f(a, n)
    b.append(5)
    return len(a) * 100 + k * 10 + a[-1]
# INPUTS [{'n': 1}, {'n': 3}]
