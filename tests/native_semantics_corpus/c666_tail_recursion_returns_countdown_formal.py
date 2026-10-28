# A directly tail-recursive callee returns its own formal.
def retry(v: int) -> int:
    if v <= 0:
        return v
    return retry(v - 1)


def root(n: int) -> int:
    return retry(n) + 1
# INPUTS [{'n': 0}, {'n': 7}, {'n': -4}]
