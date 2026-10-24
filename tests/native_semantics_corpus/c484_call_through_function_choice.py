def inc(x):
    return x + 1


def neg(x):
    return -x


def root(a: int, i: int) -> int:
    # A call through a local name that selects among module functions.
    f = inc if i == 0 else neg
    if i > 1:
        g = neg
    else:
        g = inc
    return f(a) * 100 + g(a) * 10 + (neg if a > 0 else inc)(i)

# INPUTS [{'a': 3, 'i': 0}, {'a': 7, 'i': 1}, {'a': -4, 'i': 2}, {'a': 0, 'i': 0}]
