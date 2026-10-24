def root(n: int) -> int:
    # The delete runs before the membership test that guards the branch,
    # even when the tested key is computed inline.
    d1 = {1: n, 2: 3, 3: -1}
    del d1[1]
    g1 = d1[n % 4] if n % 4 in d1 else -5
    d2 = {0: 7, 1: n, 2: 4}
    del d2[n % 3]
    g2 = d2[(n + 1) % 3] if (n + 1) % 3 in d2 else 11
    return len(d1) * 1000 + g1 * 10 + len(d2) * 100000 + g2

# INPUTS [{'n': -3}, {'n': 0}, {'n': 1}, {'n': 2}, {'n': 5}, {'n': 6}]
