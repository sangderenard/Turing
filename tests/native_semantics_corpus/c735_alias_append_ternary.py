def root(n: int) -> int:
    a = [1, 2]
    t = a if n > 2 else [3, 4, 5]
    t.append(9)
    return len(a) * 100 + len(t) + t[-1]
# INPUTS [{'n': 1}, {'n': 3}, {'n': 9}]
