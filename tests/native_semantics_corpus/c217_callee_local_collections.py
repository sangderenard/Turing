def with_list(n: int) -> int:
    xs = [7, 5]
    xs.append(n)
    return xs[0] * 10 + xs[2]


def with_dict(n: int) -> int:
    d = {1: n, 2: 5}
    d[3] = n * 2
    return d[1] * 10 + d[2] + d[3] * 100


def root(n: int) -> int:
    return with_list(n) + with_dict(n) * 1000

# INPUTS [{'n': 3}, {'n': -1}, {'n': 0}]
