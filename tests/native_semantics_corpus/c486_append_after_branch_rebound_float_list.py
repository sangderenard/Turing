def root(k: int) -> float:
    # Lane D's float spelling of c485.
    rep = [1.5]
    if k > 2:
        rep = [k * 0.5, 2.0]
    rep.append(0.25)
    return sum(rep) * 10 + len(rep) + max(rep)

# INPUTS [{'k': 0}, {'k': 3}, {'k': 5}]
