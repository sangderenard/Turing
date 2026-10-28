def root(n: int) -> int:
    values = [i % 3 for i in range(n)]
    seen = set(values)
    return len(values) * 10 + len(seen)

# INPUTS [{'n': 0}, {'n': 1}, {'n': 2}, {'n': 7}]
