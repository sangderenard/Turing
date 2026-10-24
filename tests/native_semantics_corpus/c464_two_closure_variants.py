def pick(k: int, flag: int):
    scale = k * 2
    if flag > 0:
        start = k + 1

        def finalize(x: int) -> int:
            return x * scale + start

        return finalize
    return lambda x: x


def root(n: int) -> int:
    f = pick(n, n % 2)
    total = f(3)
    return total + pick(n, 1)(1)
# INPUTS [{"n":0},{"n":3},{"n":4}]
