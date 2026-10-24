def root(k: int) -> int:
    # A list rebound to a longer display in one arm, then appended: the
    # arena holds the arm's rows plus the append, and sum/max/min read the
    # live length.
    rep = [1]
    if k > 2:
        rep = [k, 2]
    rep.append(3)
    return sum(rep) * 1000 + max(rep) * 100 + min(rep) * 10 + len(rep)

# INPUTS [{'k': 0}, {'k': 3}, {'k': 5}, {'k': -2}]
