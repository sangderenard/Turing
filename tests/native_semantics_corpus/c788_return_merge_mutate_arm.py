def pick(a: list[int], n: int) -> list[int]:
    if n > 2:
        t = a
    else:
        t = [7, n, 2]
    return t


def root(n: int) -> int:
    a = [n, 1]
    t = pick(a, n)
    a.append(5)
    return len(a) * 100 + len(t) * 10 + t[-1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
