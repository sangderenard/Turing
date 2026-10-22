def root(n: int) -> int:
    ss = {4, 6}
    return (1 if n not in ss else 0) + (2 if n in ss else 0)

# INPUTS [{'n': 1}, {'n': 4}, {'n': 6}]
