def root(n: int) -> int:
    values = [i % 3 for i in range(n)]
    seen = set(v for v in values)
    return len(seen)

# INPUTS [{'n': 0}, {'n': 2}, {'n': 7}]
