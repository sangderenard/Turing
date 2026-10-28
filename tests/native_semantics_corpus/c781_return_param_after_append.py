def grow(a: list[int], v: int) -> list[int]:
    a.append(v)
    return a


def root(n: int) -> int:
    b = [3, 4]
    t = grow(b, n)
    b.append(1)
    return len(t) * 100 + len(b) * 10 + t[-1] + t[2]
# INPUTS [{'n': 1}, {'n': 9}]
