def root(n: int) -> int:
    if n > 2: t = [n, 1]
    else: t = []
    return len(t) * 100 + (t[0] if len(t) else 0)
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
