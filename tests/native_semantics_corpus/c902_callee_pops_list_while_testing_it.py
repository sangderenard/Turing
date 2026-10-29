# A callee that pops its (unannotated) list formal in a ``while`` whose test
# reads that list, then stores into it under a test of its length.  Lowering
# refused it: the ``if``'s predicate region sat after the conditional, so the
# arena walk ordered ``len(out)`` after the arm's store while the predicate
# value ordered it before; the latch re-tested with the entry test's stale
# ``len(out)`` loads; and the unannotated formal's ``pop`` stayed opaque.
def _trim(out):
    while len(out) > 1 and out[len(out) - 1] == 0:
        out.pop()
    if len(out) == 1:
        out[0] = 0


def root(n: int):
    out = [1, n, 0, 0]
    _trim(out)
    return len(out) + out[0]

# INPUTS [{'n': 0}, {'n': 5}, {'n': -2}]
