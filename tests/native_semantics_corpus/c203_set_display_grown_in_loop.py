def root(n: int) -> int:
    s = {1, 2}
    for i in range(n % 5):
        s.add(i * 3)
    return len(s)

# INPUTS [{'n': 1}, {'n': 5}, {'n': 3}]
