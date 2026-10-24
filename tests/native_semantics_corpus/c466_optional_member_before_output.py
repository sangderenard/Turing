def pick(k: int, flag: int):
    if flag > 0:
        return (k, 0)
    return (None, flag)


def root(n: int) -> int:
    f = pick(n, n % 3)
    if f[0] is None:
        return f[1] * 10
    return f[0] + f[1]
# INPUTS [{"n":0},{"n":3},{"n":4},{"n":5}]
