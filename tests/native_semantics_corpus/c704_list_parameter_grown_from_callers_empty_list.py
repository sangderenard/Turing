# An unannotated parameter appended to by its callee is the caller's empty
# list: its length and last row are read back in the callee.
def put(xs, t: int) -> int:
    xs.append(t)
    return xs[len(xs) - 1] * 10 + len(xs)


def root(t: int) -> int:
    b = []
    put(b, t)
    return put(b, t + 1) * 100 + len(b)
# INPUTS [{'t': 5}, {'t': -3}]
