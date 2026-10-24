# Lane E (E22b): the replication's region keeps only its fill (its count
# comes from another region), so it has no body.  The control still called
# ``numerical_region_N``, which C never defined.
def root(n: int) -> int:
    xs1 = []
    for k1 in range(n % 4 + 5):
        xs1.append((k1 * -2 + n) % 9)
    ys1 = sorted(xs1)
    xs2 = [6] * (n % 4 + 1)
    return ys1[0] * 100 + ys1[-1] * 10 + len(xs1) + len(xs2) * 10 + xs2[0]

# INPUTS [{'n': 0}, {'n': 3}, {'n': 6}]
