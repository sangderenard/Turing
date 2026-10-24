def root(n: int) -> int:
    # A loop that fills a set from a list reads that list, so it runs after
    # the list is built even when one region holds an earlier store's sum
    # and both set lengths.
    xs0 = [6, n]
    xs0[1] = 1
    xs0[0] += n
    xs2 = [n % 3, 3]
    s2 = set(xs2)
    t2 = set(xs2)
    return (xs0[0] + xs0[1]) + (len(s2) * 10 + len(t2))

# INPUTS [{'n': -3}, {'n': 0}, {'n': 2}, {'n': 9}]
