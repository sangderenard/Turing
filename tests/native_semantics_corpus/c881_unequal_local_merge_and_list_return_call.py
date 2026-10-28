def mk(n: int) -> list[int]:
    return [n, n + 1, 3]


def g(n: int) -> int:
    x = [1, 2] if n > 2 else [3, 4, 5]
    x.append(n)
    return len(x)


def root(n: int) -> int:
    t = mk(n)
    return len(t) + g(n) + t[1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
