def root(n: int) -> int:
    s = {1, 2}
    s.add(n)
    return len(s) * 10 + (1 if n in s else 0)

# INPUTS [{'n': 1}, {'n': 5}, {'n': 4}]
