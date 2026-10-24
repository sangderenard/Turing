def root(n: int) -> int:
    values = [i % 3 for i in range(n)]
    seen = frozenset(values)
    return len(seen)

# INPUTS [{'n': 0}, {'n': 2}, {'n': 7}]
