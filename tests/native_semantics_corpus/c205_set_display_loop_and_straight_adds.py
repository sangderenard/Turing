def root(n: int) -> int:
    s = {1, 2}
    for i in range(n % 5):
        s.add(i * 3)
    s.add(n)
    return len(s) * 10 + (1 if 6 in s else 0) + (2 if n in s else 0)

# INPUTS [{'n': 1}, {'n': 5}, {'n': 3}, {'n': 4}]
