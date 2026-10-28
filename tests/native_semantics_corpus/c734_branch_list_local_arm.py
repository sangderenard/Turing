def root(n: int) -> int:
    other = []
    for i in range(n):
        other.append(i + n)
    other.append(7)
    if n > 2: t = other
    else: t = [n, 1]
    return len(t) * 100 + t[0] + t[len(t) - 1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
