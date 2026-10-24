def root(n: int) -> int:
    values = [i % 3 for i in range(n)]
    seen = frozenset(int(v) for v in values)
    return len(seen) * 100 + (1 if 2 in seen else 0)

# INPUTS [{'n': 0}, {'n': 1}, {'n': 2}, {'n': 7}]
