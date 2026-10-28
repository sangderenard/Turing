def mk(a: list[int]) -> list[int]:
    return a


def root(n: int) -> int:
    b = [3, 4, n]
    t = mk(b)
    b.append(5)
    t.append(6)
    return len(t) * 100 + len(b) * 10 + t[-1]
# INPUTS [{'n': 1}, {'n': 9}]
