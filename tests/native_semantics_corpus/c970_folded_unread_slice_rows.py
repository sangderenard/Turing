# The tire's ``vector_cross`` indexes with ``left[..., 1]``, which builds
# ``[slice(None)] * rank``.  Structural specialization folded the
# replication and its row display to literal Constants nothing reads; the
# replication was still refused for its row operand, and the display was
# still born with a slice selector as a float64 row.
def root(n: int) -> int:
    idx = [slice(None)] * 3
    rows = [slice(None), slice(1, 2)]
    return n * 10 + len(idx) + len(rows)
# INPUTS [{'n': 0}, {'n': 4}, {'n': -7}]
