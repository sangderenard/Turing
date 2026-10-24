def pick(n: int) -> list[int]:
    if n == 0:
        return [5, 6, 7]
    if n == 1:
        return [8]
    return [n, n + 1, n + 2, n * 4]


def root(n: int) -> int:
    a = pick(n)
    b = pick(n + 1)
    return len(a) * 1000 + len(b) * 100 + a[-1] * 10 + b[0]

# INPUTS [{"n": 0}, {"n": 1}, {"n": 2}]
