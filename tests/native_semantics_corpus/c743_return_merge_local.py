def make(n: int) -> list[int]:
    a = [n, 1]
    t = a if n > 2 else [7, n, 2]
    return t


def root(n: int) -> int:
    t = make(n)
    t.append(5)
    return len(t) * 10 + t[0] + t[-1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
