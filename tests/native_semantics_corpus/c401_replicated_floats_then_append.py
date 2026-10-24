# Lane E (Fortran list growth): a runtime-count replication fills through a
# helper that returns its status from two places, then an append grows the
# arena past its first capacity.  (sum(ys) here is a separate C finding:
# the reduction reads the replicated extent and misses the appends.)
def root(n: int) -> float:
    ys = [0.5] * (n % 4)
    ys.append(n * 1.25)
    ys.append(2.0)
    t = 0.0
    for y in ys:
        t = t + y
    return len(ys) * 100.0 + t + ys[0]

# INPUTS [{'n': 0}, {'n': 1}, {'n': 3}, {'n': 6}]
