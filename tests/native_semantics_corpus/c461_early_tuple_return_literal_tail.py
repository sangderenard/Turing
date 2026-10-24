def pick(k: int, flag: int):
    if flag > 0:
        return (0, k)
    return (1, 0)


def root(n: int) -> int:
    f = pick(n, n % 2)
    return f[0] * 10 + f[1]
# INPUTS [{"n":0},{"n":3}]
