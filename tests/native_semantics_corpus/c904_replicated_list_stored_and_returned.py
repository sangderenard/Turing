# A list born in the callee by replication, stored into, then returned: each
# call returns a fresh list.  The call bound the caller's view to the store
# node instead of the list it writes, so the view named no arena (refused).
def g(n):
    r = [0] * 3
    r[1] = n
    return r


def root(n: int) -> int:
    t = g(n)
    u = g(n + 1)
    t[0] = 7
    return t[0] + t[1] * 10 + t[2] * 100 + u[0] * 1000 + u[1] * 10000 + len(t)

# INPUTS [{'n': 0}, {'n': 4}, {'n': -7}]
