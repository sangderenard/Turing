def root(n: int) -> int:
    ev = [i + 1 for i in range(n) if i % 2 == 0]
    big = any(i * i > n for i in range(n))
    allpos = all(i >= 0 for i in range(n))
    return sum(ev) * 100 + len(ev) * 10 + (1 if big else 0) + (2 if allpos else 0)

# INPUTS [{'n': 0}, {'n': 1}, {'n': 5}, {'n': 8}]
