# A callee stores into its list formal and returns it: ``t`` IS ``a``, so the
# caller's stores through either name are seen through the other.  The
# returned name was the store's version of ``out``, which no returned-formal
# reader followed back to the list; the program was refused.
def f(out, x):
    out[0] = x
    return out


def root(n: int) -> int:
    a = [1, 2, 3]
    t = f(a, n)
    t[1] = n + 5
    a[2] = n * 3
    return a[0] + a[1] * 10 + a[2] * 100 + t[0] * 1000 + t[2] * 10000

# INPUTS [{'n': 0}, {'n': 4}, {'n': -7}]
