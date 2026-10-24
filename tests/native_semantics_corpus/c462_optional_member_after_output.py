def pick(k: int, flag: int):
    if flag > 0:
        return (0, k)
    return (flag, None)


def root(n: int) -> int:
    f = pick(n, n % 2)
    if f[1] is None:
        return -1
    return f[1]
# INPUTS [{"n":0},{"n":3},{"n":4}]
