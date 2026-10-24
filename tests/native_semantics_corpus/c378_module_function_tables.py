# Module tables of known functions, indexed and called (plan C25).  The
# tables hold variant tags and each call goes through a generated
# dispatcher (closure_values.lower_known_function_selections).
def inc(x):
    return x + 1


def neg(x):
    return -x


def scale(x):
    return x * 3


OPS = {0: inc, 1: neg, 2: scale}
TABLE = [scale, inc]
PAIR = (neg, neg)


def root(a: int, i: int) -> int:
    k = i % 3
    return OPS[k](a) * 100 + TABLE[1 - i % 2](a) * 10 + PAIR[i % 2](k)
# INPUTS [{"a": 3, "i": 0}, {"a": 7, "i": 1}, {"a": -2, "i": 2}, {"a": 0, "i": 5}]
