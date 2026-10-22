def root(n: int) -> int:
    s = set()
    for i in range(n):
        s.add(i % 4)
    return len(s) * 10 + (1 if 2 in s else 0)

# INPUTS [{'n': 2}, {'n': 9}]
