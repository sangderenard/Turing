def root(n: int) -> int:
    # The region after a ``del`` stores into a list cell and publishes no
    # value; the delete runs before it and the store still runs.
    d0 = {1: 2, 2: n, 3: 2}
    del d0[2]
    xs1 = [4, n, 7]
    xs1[1] = 0
    xs1[0] += n
    return len(d0) * 1000 + xs1[0] * 100 + xs1[1] * 10 + xs1[2]

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}]
