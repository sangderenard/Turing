def root(n: int) -> int:
    if n > 2: t = [n, 1, 2]
    else: t = [7, n]
    t.append(n + 5)
    return len(t) * 100 + t[2] + t[len(t) - 1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
